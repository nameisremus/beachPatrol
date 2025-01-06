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

# tasks_list will hold tuples: (job, ctx, media_type)
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

    # Start our daily scheduled run
    daily_govdigest_scheduler.start()
    print("Daily scheduled govdigest loop started.")

@bot.slash_command()
async def generate_summary(ctx, url: str):
    """
    /generate_summary <url> -> determines media type and enqueues a summary generation task.
    """
    if "twitter.com/i/spaces" in url or "x.com/i/spaces" in url:
        await ctx.respond(f"Processing Twitter Space URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_space", args=[url])
        tasks_list.append((job, ctx, "twitter_space"))

    elif "youtube.com/watch" in url or "youtu.be" in url:
        await ctx.respond(f"Processing YouTube URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_youtube_video", args=[url])
        tasks_list.append((job, ctx, "youtube"))

    else:
        await ctx.respond(f"Processing article URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_article", args=[url])
        tasks_list.append((job, ctx, "article"))

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
    
    await ctx.respond(
        f"Processing governance forum updates... timeframe={actual_timeframe}, "
        f"onlyRelevant={actual_relevancy_filter}"
    )
    job = celery_app.send_task(
        "worker.scrape_governance_forum",
        kwargs={"timeframe": actual_timeframe, "only_relevant": actual_relevancy_filter}
    )
    tasks_list.append((job, ctx, "governance_forum"))

@tasks.loop(seconds=5)
async def check_tasks():
    for (job, ctx, media_type) in tasks_list[:]:
        if job.ready():
            result = job.get()
            print(f"[check_tasks] Media type={media_type}, result={result}")

            if not result or isinstance(result, str):
                # Means an error or "No updates." or similar
                if ctx is not None:
                    await ctx.followup.send(str(result))
                else:
                    # fallback if no ctx
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await channel.send(str(result))
                tasks_list.remove((job, ctx, media_type))
                continue

            exec_sum, notes = result
            exec_sum = exec_sum or ""
            notes = notes or ""

            # We'll store pages in 'pages'.
            pages = []

            # Mapping for base titles and disclaimers
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
                }
            }

            # Default fallback if media_type not in above
            default_config = {
                "base_title": f"Update, model={OPENAI_MODEL}",
                "disclaimers": ""
            }

            # Get the configuration for the given media_type or fallback to default
            config = media_configs.get(media_type, default_config)

            # Prepare the first_text with disclaimers
            disclaimers = config["disclaimers"]
            first_text = exec_sum + disclaimers

            # Chunk the first_text and the notes
            first_chunks = chunk_text(first_text, limit=3846)
            notes_pages = chunk_text(notes, limit=3846)

            # Combine the chunks into pages
            pages = first_chunks + notes_pages
            base_title = config["base_title"]

            # Now handle how to send the results
            if len(pages) == 0:
                if ctx:
                    await ctx.followup.send("No updates.")
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await channel.send("No updates.")

            elif len(pages) == 1:
                # Only one page -> just send an embed
                embed = discord.Embed(title=base_title, description=pages[0], color=0x2F3136)
                if ctx:
                    await ctx.followup.send(embed=embed)
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await channel.send(embed=embed)
            else:
                # Multiple pages -> Create a PersistentPaginatedEmbedView
                report_date = date.today().strftime("%Y-%m-%d")
                unique_id = job.id or "nojobid"
                report_id = f"{media_type}_{report_date}_{unique_id}"

                view = PersistentPaginatedEmbedView(
                    report_id=report_id,
                    pages=pages,
                    base_title=base_title,
                    current_index=0
                )

                # Create the first embed
                first_embed = view._get_embed()

                # Send the message
                if ctx:
                    msg = await ctx.followup.send(embed=first_embed, view=view)
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if not channel:
                        tasks_list.remove((job, ctx, media_type))
                        continue
                    msg = await channel.send(embed=first_embed, view=view)

                # Store the sent message reference in the view
                view.message = msg
                view.message_id = msg.id
                view.channel_id = msg.channel.id

                # Save the entire data in Redis
                data = {
                    "pages": pages,
                    "current_index": 0,
                    "base_title": base_title,
                    "message_id": msg.id,
                    "channel_id": msg.channel.id
                }
                r.set(f"report:{report_id}", json.dumps(data))

                # Finally add the view so it can handle future interactions
                bot.add_view(view)

            tasks_list.remove((job, ctx, media_type))

@tasks.loop(seconds=60)
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

        # Remove the item from the list in Redis after processing
        r.lrem("watchlist_results", 1, result_bytes)

@tasks.loop(minutes=1)
async def daily_govdigest_scheduler():
    """
    Runs every minute, checks if it's 10:00 UTC. If yes, 
    calls the Celery task with timeframe=1d, only_relevant=True.
    """
    now_utc = datetime.now(timezone.utc)
    # Example: check for 10:10 UTC
    if now_utc.hour == 10 and now_utc.minute == 10:
        print("[daily_govdigest_scheduler] It's 10:00 UTC -> scheduling daily govdigest.")
        # direct Celery call
        job = celery_app.send_task(
            "worker.scrape_governance_forum",
            kwargs={"timeframe": "1d", "only_relevant": True}
        )
        # we have no user context so tasks_list has (job, None, "governance_forum")
        tasks_list.append((job, None, "governance_forum"))

@daily_govdigest_scheduler.before_loop
async def before_daily_govdigest_scheduler():
    print("Waiting for bot to get ready before daily_govdigest_scheduler loop.")
    await bot.wait_until_ready()

# Start all loops
check_tasks.start()
check_watchlist_results.start()

# Run the bot
bot.run(DISCORD_TOKEN)