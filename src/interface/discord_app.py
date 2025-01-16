import discord
from discord.ext import tasks
from discord import Intents, Option
from dotenv import load_dotenv
import json
import redis

from datetime import date, datetime, timezone

from config import (
    DISCORD_TOKEN, DISCORD_CHANNEL_ID, REDIS_HOST, REDIS_PORT,
    OPENAI_MODEL, OPENAI_GOVERNANCE_MODEL
)
from tasks.celery_config import app as celery_app
from core.utils import chunk_text, PersistentPaginatedEmbedView

load_dotenv()

intents = Intents.default()
intents.presences = True
bot = discord.Bot(intents=intents, command_prefix="/")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)

# tasks_list will hold tuples: (job, ctx, media_type, msg_id, last_update_count)
#
# Where:
#   - job: Celery AsyncResult
#   - ctx: The Discord context (for followups)
#   - media_type: e.g. "twitter_digest"
#   - msg_id: The message ID for the "Processing..." message we want to edit
#   - last_update_count: The last processed count we displayed (so we don't spam updates)
tasks_list = []

@bot.event
async def on_ready():
    print("Logged in as " + bot.user.name)

    # Re-register existing daily/persistent reports from Redis
    for key_bytes in r.scan_iter("report:*"):
        key_str = key_bytes.decode("utf-8")
        data_json = r.get(key_str)
        if not data_json:
            continue
        try:
            data = json.loads(data_json)
            pages = data["pages"]
            current_index = data["current_index"]
            base_title = data["base_title"]
            message_id = data.get("message_id")
            channel_id = data.get("channel_id")
            
            report_id = key_str.removeprefix("report:")

            # Create the view object from the saved data
            view = PersistentPaginatedEmbedView(
                report_id=report_id,
                pages=pages,
                base_title=base_title,
                current_index=current_index,
                message_id=message_id,
                channel_id=channel_id
            )

            # Attempt to fetch the original message if we have IDs
            if message_id and channel_id:
                channel = bot.get_channel(channel_id)
                if channel:
                    try:
                        msg = await channel.fetch_message(message_id)
                        view.message = msg
                    except discord.NotFound:
                        print(f"Message {message_id} not found in channel {channel_id}")
                    except Exception as e:
                        print(f"Error fetching message: {e}")

            # Now add the persistent view so it can receive interactions
            bot.add_view(view)
            print(f"Re-registered persistent view for {report_id} with {len(pages)} pages.")
        except Exception as e:
            print(f"Error re-initializing persistent view for key {key_str}: {e}")

    # Start the daily scheduled tasks loop for gov and twitter digests
    daily_scheduled_tasks.start()
    print("Daily scheduled tasks loop started.")

@bot.slash_command(description="Check if the bot is responsive.")
async def ping(ctx):
    """
    /ping -> Responds with 'Pong!' and the bot's latency.
    """
    await ctx.respond(f"Pong! 🏓")

@bot.slash_command()
async def generate_summary(ctx, url: str):
    """
    /generate_summary <url> -> determines media type and enqueues a summary generation task.
    """
    if "twitter.com/i/spaces" in url or "x.com/i/spaces" in url:
        await ctx.respond(f"Processing Twitter Space URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_space", args=[url])
        tasks_list.append((job, ctx, "twitter_space", None, 0))

    elif "youtube.com/watch" in url or "youtu.be" in url:
        await ctx.respond(f"Processing YouTube URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_youtube_video", args=[url])
        tasks_list.append((job, ctx, "youtube", None, 0))

    else:
        await ctx.respond(f"Processing article URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_article", args=[url])
        tasks_list.append((job, ctx, "article", None, 0))

@bot.slash_command(description="Manually run governance forum scraping with optional timeframe and relevancy.")
async def generate_gov_digest(
    ctx,
    timeframe: Option(str, "Timeframe for topics (e.g., 1d, 7d, 30d)", default="1d"),  # type: ignore
    relevancy_filter: Option(bool, "Only include topics relevant to Lido/LSTs/eco/ETH", default=True)  # type: ignore
):
    """
    /generate_gov_digest [timeframe=7d] [relevancy_filter=True]
    Args:
        timeframe (str): 1d/7d/etc - timeframe for topics
        relevancy_filter (bool): True/False - if you only want topics relevant to Lido/LSTs/Eco/ETH
    """
    # Extract the actual values
    actual_timeframe = str(timeframe)
    actual_relevancy_filter = bool(relevancy_filter)
    
    # Send an initial response to Discord
    original_msg = await ctx.respond(
        f"Processing governance forum updates... timeframe={actual_timeframe}, "
        f"onlyRelevant={actual_relevancy_filter}"
    )
    # Get the actual message we just sent
    sent_msg = await ctx.interaction.original_response()

    # Kick off the Celery job
    job = celery_app.send_task(
        "worker.scrape_governance_forum",
        kwargs={"timeframe": actual_timeframe, "only_relevant": actual_relevancy_filter}
    )

    # Store references to this job and the message
    tasks_list.append((job, ctx, "governance_forum", sent_msg.id, 0))

@bot.slash_command(description="Run a Twitter digest for over 250 usernames, optional timeframe & relevancy.")
async def generate_twitter_digest(
    ctx,
    timeframe: Option(str, "Timeframe for tweets ('1d' or '2d' maximum due to rate limits)", default="1d"), # type: ignore
    relevancy_filter: Option(bool, "Only include tweets relevant to Lido/ETH, etc.?", default=True)  # type: ignore
):
    """
    /generate_twitter_digest [timeframe=1d or 2d] [relevancy_filter=True]
    Gathers tweets from the configured usernames, optionally filtering for Lido/ETH relevancy.
    """
    # 1. Check if timeframe is allowed
    if timeframe not in ("1d", "2d"):
        await ctx.respond(
            "Due to Twitter rate limits, the timeframe can only be **1d** or **2d**. "
            "Please try again with a valid timeframe."
        )
        return

    # 2. If valid timeframe, proceed with Celery task
    original_msg = await ctx.respond(
        f"Processing Twitter digest... timeframe={timeframe}, onlyRelevant={relevancy_filter}"
    )
    # Fetch the actual message we sent
    sent_msg = await ctx.interaction.original_response()

    job = celery_app.send_task(
        "worker.scrape_twitter_digest",
        kwargs={"timeframe": timeframe, "only_relevant": relevancy_filter}
    )
    # tasks_list gets the job & the message ID
    tasks_list.append((job, ctx, "twitter_digest", sent_msg.id, 0))

@bot.slash_command(description="Summarize tweets for one Twitter user, optional timeframe.")
async def generate_twitter_account_summary(
    ctx,
    username: Option(str, "The Twitter username (no @)", default="elonmusk"),  # type: ignore
    timeframe: Option(str, "Timeframe for tweets (e.g. 1d, 7d, 30d)", default="1d")  # type: ignore
):
    """
    /generate_twitter_account_summary <username> [timeframe=1d]
    Summarizes all tweets for that user within the timeframe, no relevancy filter.
    """
    # Acknowledge
    original_msg = await ctx.respond(
        f"Processing Twitter summary for @{username}, timeframe={timeframe}..."
    )
    sent_msg = await ctx.interaction.original_response()

    job = celery_app.send_task(
        "worker.scrape_twitter_account_summary",
        kwargs={"username": username, "timeframe": timeframe}
    )

    # Add reference so we can watch progress in check_tasks
    tasks_list.append((job, ctx, "single_twitter_account", sent_msg.id, 0))

@tasks.loop(seconds=5)
async def check_tasks():
    """
    Every 5 seconds, we:
    - Check if a job is complete (job.ready()). If so, retrieve final result.
    - If it's still in progress (job.state == "PROGRESS"), we read job.info to see how many accounts processed.
    - Update the original Discord message accordingly.
    """
    for (job, ctx, media_type, msg_id, last_count) in tasks_list[:]:
        if job.ready():
            # Final result
            result = job.get()
            print(f"[check_tasks] Media type={media_type}, result={result}")

            # If result is an error or simple string
            if not result or isinstance(result, str):
                if ctx is not None and msg_id:
                    try:
                        channel = ctx.channel
                        msg_to_edit = await channel.fetch_message(msg_id)
                        await msg_to_edit.edit(content=str(result) or "No updates.")
                    except Exception:
                        pass
                    # Send result directly via channel to avoid expired interactions
                    await ctx.channel.send(str(result))
                else:
                    # fallback if no ctx
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await channel.send(str(result))

                tasks_list.remove((job, ctx, media_type, msg_id, last_count))
                continue

            # Otherwise it's a tuple: (exec_sum, notes)
            exec_sum, notes = result
            exec_sum = exec_sum or ""
            notes = notes or ""

            # Edit original message to final
            if ctx is not None and msg_id:
                try:
                    channel = ctx.channel
                    msg_to_edit = await channel.fetch_message(msg_id)
                    await msg_to_edit.edit(
                        content=(
                            f"Processing {media_type} finished!\n"
                            "Command has been processed."
                        )
                    )
                except Exception as e:
                    print(f"[check_tasks] Error editing final message: {e}")

            # Build final response
            pages = []
            media_configs = {
                "governance_forum": {
                    "base_title": "Ecosystem and Governance Updates",
                    "disclaimers": (
                        f"\n\n*(This summary was generated by a refined model. The following pages "
                        f"contain a more comprehensive summary using: **{OPENAI_MODEL}**.)*\n\n"
                    )
                },
                "article": {
                    "base_title": f"Article Summary, model={OPENAI_MODEL}",
                    "disclaimers": (
                        "\n\n*(Above is the Executive Summary. Next pages contain a more "
                        f"comprehensive summary.)*\n\n"
                    )
                },
                "youtube": {
                    "base_title": f"YouTube Video Summary, model={OPENAI_MODEL}",
                    "disclaimers": (
                        "\n\n*(Above is the Executive Summary. Next pages contain a more "
                        f"comprehensive summary.)*\n\n"
                    )
                },
                "twitter_space": {
                    "base_title": f"Twitter Space Summary, model={OPENAI_MODEL}",
                    "disclaimers": (
                        "\n\n*(Above is the Executive Summary. Next pages contain a more "
                        f"comprehensive summary.)*\n\n"
                    )
                },
                "twitter_digest": {
                    "base_title": f"Twitter Digest, model={OPENAI_MODEL}",
                    "disclaimers": (
                        "\n\n*(Above is the Executive Summary. Next pages contain a more "
                        f"comprehensive summary.)*\n\n"
                    )
                },
                "single_twitter_account": {
                    "base_title": f"Single Twitter Account Summary, model={OPENAI_MODEL}",
                    "disclaimers": (
                        "\n\n*(Above is the Executive Summary. Next pages contain a more "
                        "comprehensive summary of tweets for this user.)*\n\n"
                    )
                },
            }
            default_config = {
                "base_title": f"Update, model={OPENAI_MODEL}",
                "disclaimers": ""
            }
            config = media_configs.get(media_type, default_config)
            disclaimers = config["disclaimers"]
            first_text = exec_sum + disclaimers

            first_chunks = chunk_text(first_text, limit=3846)
            notes_pages = chunk_text(notes, limit=3846)
            pages = first_chunks + notes_pages
            base_title = config["base_title"]

            # Send final result
            if len(pages) == 0:
                if ctx:
                    await ctx.channel.send("No updates.")
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await channel.send("No updates.")
            elif len(pages) == 1:
                embed = discord.Embed(title=base_title, description=pages[0], color=0x2F3136)
                if ctx:
                    await ctx.channel.send(embed=embed)
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await channel.send(embed=embed)
            else:
                report_date = date.today().strftime("%Y-%m-%d")
                unique_id = job.id or "nojobid"
                report_id = f"{media_type}_{report_date}_{unique_id}"

                # only truncate if media_type == 'single_twitter_account'
                if media_type == "single_twitter_account":
                    report_id = report_id[:50]

                view = PersistentPaginatedEmbedView(
                    report_id=report_id,
                    pages=pages,
                    base_title=base_title,
                    current_index=0
                )

                first_embed = view._get_embed()
                if ctx:
                    msg = await ctx.channel.send(embed=first_embed, view=view)
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if not channel:
                        tasks_list.remove((job, ctx, media_type, msg_id, last_count))
                        continue
                    msg = await channel.send(embed=first_embed, view=view)

                view.message = msg
                view.message_id = msg.id
                view.channel_id = msg.channel.id

                data = {
                    "pages": pages,
                    "current_index": 0,
                    "base_title": base_title,
                    "message_id": msg.id,
                    "channel_id": msg.channel.id
                }
                r.set(f"report:{report_id}", json.dumps(data))
                bot.add_view(view)

            tasks_list.remove((job, ctx, media_type, msg_id, last_count))

        else:
            # Task is still running, check for "PROGRESS"
            try:
                state = job.state
                info = job.info or {}
                if state == "PROGRESS":
                    processed = info.get("processed", 0)
                    total = info.get("total", 0)
                    timeframe = info.get("timeframe", "")
                    only_rel = info.get("only_relevant", "")

                    if (processed - last_count) >= 10:
                        new_content = (
                            f"**:hourglass_flowing_sand: Processing {media_type}...**\n"
                            f"> Processed **{processed}** / **{total}** accounts\n"
                            f"> Timeframe: `{timeframe}`\n"
                            f"> Relevancy filter: `{only_rel}`"
                        )
                        
                        if ctx is not None and msg_id:
                            channel = ctx.channel
                            try:
                                msg_to_edit = await channel.fetch_message(msg_id)
                                await msg_to_edit.edit(content=new_content)
                            except Exception as e:
                                print(f"[check_tasks] Error editing message: {e}")

                        idx = tasks_list.index((job, ctx, media_type, msg_id, last_count))
                        tasks_list[idx] = (job, ctx, media_type, msg_id, processed)
            except Exception as e:
                print(f"[check_tasks] Exception checking task progress: {e}")

@tasks.loop(minutes=1)
async def daily_scheduled_tasks():
    """
    Runs every minute, scheduling:
      - Governance digest at e.g. 10:00 UTC
      - Twitter digest at e.g. 10:05 UTC
    """
    now_utc = datetime.now(timezone.utc)

    # 1) If it's 10:00, schedule governance digest
    if now_utc.hour == 10 and now_utc.minute == 0:
        print("[daily_scheduled_tasks] It's 10:00 UTC -> scheduling daily govdigest.")
        job = celery_app.send_task(
            "worker.scrape_governance_forum",
            kwargs={"timeframe": "1d", "only_relevant": True}
        )
        tasks_list.append((job, None, "governance_forum", None, 0))

    # 2) If it's 06:00, schedule Twitter digest
    if now_utc.hour == 6 and now_utc.minute == 0:
        print("[daily_scheduled_tasks] It's 07:00 UTC -> scheduling daily twitterdigest.")
        job2 = celery_app.send_task(
            "worker.scrape_twitter_digest",
            kwargs={"timeframe": "1d", "only_relevant": True}
        )
        tasks_list.append((job2, None, "twitter_digest", None, 0))


@daily_scheduled_tasks.before_loop
async def before_daily_scheduled_tasks():
    print("Waiting for bot to get ready before daily_scheduled_tasks loop.")
    await bot.wait_until_ready()
    print("Ready.")

@tasks.loop(seconds=5)
async def check_watchlist_results():
    """
    Periodically checks a Redis list for new watchlist results 
    and sends them to the specified channel.
    """
    channel = bot.get_channel(DISCORD_CHANNEL_ID)
    for result_bytes in r.lrange("watchlist_results", 0, -1):
        result_str = result_bytes.decode("utf-8")
        print(result_str)
        result_dict = json.loads(result_str)

        space_url = result_dict["space_url"]
        exec_sum = result_dict["exec_sum"] or ""
        notes = result_dict["notes"] or ""

        if channel:
            combined_text = (
                f"**Twitter Space**: {space_url}\n\n"
                f"**Short Summary**:\n{exec_sum}\n\n"
                f"**Full Summary**:\n{notes}"
            )
            pages = chunk_text(combined_text, limit=3846)
            if len(pages) == 1:
                embed = discord.Embed(
                    title="Watchlist Twitter Space",
                    description=pages[0],
                    color=0x2F3136
                )
                await channel.send(embed=embed)
            else:
                for i, chunk in enumerate(pages):
                    embed = discord.Embed(
                        title=f"Watchlist Twitter Space (page {i+1}/{len(pages)})",
                        description=chunk,
                        color=0x2F3136
                    )
                    await channel.send(embed=embed)

        r.lrem("watchlist_results", 1, result_bytes)

# Start loops that are not started in on_ready()
check_tasks.start()
check_watchlist_results.start()
# daily_scheduled_tasks is started in on_ready()

bot.run(DISCORD_TOKEN)