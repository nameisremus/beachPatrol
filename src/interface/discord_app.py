import discord
from discord.ext import tasks
from discord import Intents, Option
from dotenv import load_dotenv
import json
import redis

from datetime import date, datetime, timezone

from config import (
    DISCORD_TOKEN, DISCORD_CHANNEL_ID, REDIS_HOST, REDIS_PORT,
    OPENAI_MODELS_LIST, OPENAI_MODEL,
    BOT_PASSWORD
)
from tasks.celery_config import app as celery_app
from core.utils import chunk_text, PersistentPaginatedEmbedView
from core.media_configs import MEDIA_CONFIGS, DEFAULT_MEDIA_CONFIG
from core.core import get_content_tags

load_dotenv()

intents = Intents.default()
intents.presences = True
bot = discord.Bot(intents=intents, command_prefix="/")

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT)
tasks_list = []


def is_guild_authorized(guild_id: int) -> bool:
    return r.sismember("authorized_discord_guilds", str(guild_id))

@bot.event
async def on_ready():
    print("Logged in as " + bot.user.name)

    # Re-register persistent views
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

            view = PersistentPaginatedEmbedView(
                report_id=report_id,
                pages=pages,
                base_title=base_title,
                current_index=current_index,
                message_id=message_id,
                channel_id=channel_id
            )
            view.notion_status = data.get("notion_status", "default")
            view.notion_executive = data.get("notion_executive", "")
            view.notion_summary = data.get("notion_summary", "")
            view.notion_tags = data.get("notion_tags", [])
            view.notion_url = data.get("notion_url", "")
            view.notion_media_type = data.get("notion_media_type", "")

            if message_id and channel_id:
                channel = bot.get_channel(channel_id)
                if channel:
                    try:
                        msg = await channel.fetch_message(message_id)
                        view.message = msg
                        await msg.edit(embed=view._get_embed(), view=view)
                    except discord.NotFound:
                        print(f"Message {message_id} not found in channel {channel_id}")
                    except Exception as e:
                        print(f"Error fetching message: {e}")

            bot.add_view(view)
            print(f"Re-registered persistent view for {report_id} with {len(pages)} pages.")
        except Exception as e:
            print(f"Error re-initializing persistent view for key {key_str}: {e}")

    # Start the daily scheduled tasks loop for gov and twitter digests
    daily_scheduled_tasks.start()
    print("Daily scheduled tasks loop started.")

@bot.slash_command(description="Enter the bot password to unlock commands.")
async def password(ctx, password: str):
    if password == BOT_PASSWORD:
        r.sadd("authorized_discord_guilds", str(ctx.guild_id))
        await ctx.respond("✅ Bot commands now unlocked!")
    else:
        await ctx.respond("❌ Incorrect password. Please try again.")


@bot.slash_command(description="Check if the bot is responsive.")
async def ping(ctx):
    """
    /ping -> Responds with 'Pong!'.
    """
    if not is_guild_authorized(ctx.guild_id):
        await ctx.respond("This server must be unlocked first. Use `/password <BOT_PASSWORD>`.")
        return

    await ctx.respond(f"Pong! 🏓")

@bot.slash_command(description="List all available models.")
async def list_available_models(ctx):
    """
    /list_available_models -> Lists the available models
    """
    if not is_guild_authorized(ctx.guild_id):
        await ctx.respond("This server must be unlocked first. Use `/password <BOT_PASSWORD>`.")
        return

    if not OPENAI_MODELS_LIST or not any(m.strip() for m in OPENAI_MODELS_LIST):
        await ctx.respond("No models available (OPENAI_MODELS is empty).")
        return
    msg = "**Available Models:**\n"
    for m in OPENAI_MODELS_LIST:
        if m.strip():
            msg += f"- {m.strip()}\n"
    await ctx.respond(msg)

@bot.slash_command()
async def generate_summary(
    ctx,
    url: Option(str, "URL to summarize"), # type: ignore
    model: Option(str, "OpenAI model to use (optional)", default=None),  # type: ignore
    prompt: Option(str, "Custom prompt instead of default", default=None)  # type: ignore
):
    """
    /generate_summary <url> -> Summarize the URL, optionally override model and prompt.
    """
    if not is_guild_authorized(ctx.guild_id):
        await ctx.respond("This server must be unlocked first. Use `/password <BOT_PASSWORD>`.")
        return

    if "twitter.com/i/spaces" in url or "x.com/i/spaces" in url:
        await ctx.respond(f"Processing Twitter Space URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_space", args=[url], kwargs={"model": model, "prompt": prompt})
        tasks_list.append((job, ctx, "twitter_space", None, 0, url))

    elif "youtube.com/watch" in url or "youtu.be" in url:
        await ctx.respond(f"Processing YouTube URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_youtube_video", args=[url], kwargs={"model": model, "prompt": prompt})
        tasks_list.append((job, ctx, "youtube", None, 0, url))
    elif ("twitter.com/" in url or "x.com/" in url) and "/status/" in url:
        await ctx.respond(f"Processing tweet summary for URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_tweet_summary", kwargs={
            "url": url,
            "parse_comments": False,
            "model": model,
            "prompt": prompt
        })
        tasks_list.append((job, ctx, "tweet_summary", None, 0, url))
    else:
        await ctx.respond(f"Processing URL {url} ... please wait")
        job = celery_app.send_task("worker.scrape_article", args=[url], kwargs={"model": model, "prompt": prompt})
        tasks_list.append((job, ctx, "article", None, 0, url))

@bot.slash_command(description="Manually run governance forum scraping")
async def generate_gov_digest(
    ctx,
    timeframe: Option(str, "Timeframe for topics, e.g. 1d, 2d etc.", default="1d"),  # type: ignore
    relevancy_filter: Option(bool, "Only relevant topics?", default=True)  # type: ignore
):
    """
    /generate_gov_digest [timeframe=7d] [relevancy_filter=True]
    Args:
        timeframe (str): 1d/7d/etc - timeframe for topics
        relevancy_filter (bool): True/False - if you only want topics relevant to Lido/LSTs/Eco/ETH
    """
    if not is_guild_authorized(ctx.guild_id):
        await ctx.respond("This server must be unlocked first. Use `/password <BOT_PASSWORD>`.")
        return

    original_msg = await ctx.respond(
        f"Processing governance forum updates... timeframe={timeframe}, relevancyFilter={relevancy_filter}"
    )
    sent_msg = await ctx.interaction.original_response()

    job = celery_app.send_task(
        "worker.scrape_governance_forum",
        kwargs={"timeframe": timeframe, "only_relevant": relevancy_filter}
    )
    tasks_list.append((job, ctx, "governance_forum", sent_msg.id, 0, ""))

@bot.slash_command(description="Run a Twitter digest for multiple accounts.")
async def generate_twitter_digest(
    ctx,
    timeframe: Option(str, "Timeframe for tweets (1d or 2d)", default="1d"),  # type: ignore
    relevancy_filter: Option(bool, "Only relevant tweets?", default=True)  # type: ignore
):
    """
    /generate_twitter_digest [timeframe=1d or 2d] [relevancy_filter=True]
    Gathers tweets from the configured usernames, optionally filtering for Lido/ETH relevancy.
    """
    if not is_guild_authorized(ctx.guild_id):
        await ctx.respond("This server must be unlocked first. Use `/password <BOT_PASSWORD>`.")
        return

    if timeframe not in ("1d", "2d"):
        await ctx.respond(
            "Due to Twitter rate limits, the timeframe can only be **1d** or **2d**."
        )
        return
    original_msg = await ctx.respond(
        f"Processing Twitter digest... timeframe={timeframe}, relevancyFilter={relevancy_filter}"
    )
    sent_msg = await ctx.interaction.original_response()

    job = celery_app.send_task(
        "worker.scrape_twitter_digest",
        kwargs={"timeframe": timeframe, "only_relevant": relevancy_filter}
    )
    tasks_list.append((job, ctx, "twitter_digest", sent_msg.id, 0, ""))

@bot.slash_command(description="Summarize tweets for one user, optional timeframe.")
async def generate_twitter_account_summary(
    ctx,
    username: Option(str, "Twitter username (no @)", default="elonmusk"),  # type: ignore
    timeframe: Option(str, "Timeframe, e.g. 1d", default="1d"),  # type: ignore
    model: Option(str, "OpenAI model to use (optional)", default=None),  # type: ignore
    prompt: Option(str, "Custom prompt instead of default", default=None)  # type: ignore
):
    if not is_guild_authorized(ctx.guild_id):
        await ctx.respond("This server must be unlocked first. Use `/password <BOT_PASSWORD>`.")
        return

    original_msg = await ctx.respond(
        f"Processing Twitter summary for @{username}, timeframe={timeframe}..."
    )
    sent_msg = await ctx.interaction.original_response()

    job = celery_app.send_task(
        "worker.scrape_twitter_account_summary",
        kwargs={"username": username, "timeframe": timeframe, "model": model, "prompt": prompt}
    )
    tasks_list.append((job, ctx, "single_twitter_account", sent_msg.id, 0, ""))

@bot.slash_command(description="Summarize a tweet (or thread) with optional parsing of comments.")
async def generate_tweet_summary(
    ctx,
    url: Option(str, "Tweet URL"), # type: ignore
    parse_comments: Option(bool, "Parse tweet comments?", default=False),  # type: ignore
    model: Option(str, "OpenAI model to use (optional)", default=None),  # type: ignore
    prompt: Option(str, "Custom prompt instead of default", default=None)  # type: ignore
):
    if not is_guild_authorized(ctx.guild_id):
        await ctx.respond("This server must be unlocked first. Use `/password <BOT_PASSWORD>`.")
        return

    original_msg = await ctx.respond(
        f"Processing tweet summary for URL {url}, parse_comments={parse_comments}..."
    )
    sent_msg = await ctx.interaction.original_response()

    job = celery_app.send_task(
        "worker.scrape_tweet_summary",
        kwargs={
            "url": url,
            "parse_comments": parse_comments,
            "model": model,
            "prompt": prompt
        }
    )
    tasks_list.append((job, ctx, "tweet_summary", sent_msg.id, 0, url))

@tasks.loop(seconds=5)
async def check_tasks():
    """
    Every 5 seconds, we:
    - Check if a job is complete (job.ready()). If so, retrieve final result.
    - If still "PROGRESS", we read job.info to see how many accounts processed.
    - Always create a PersistentPaginatedEmbedView (so user sees same buttons even if single page).
    """
    for (job, ctx, media_type, msg_id, last_count, url_for_content) in tasks_list[:]:
        if job.ready():
            result = job.get()

            # If result is empty or a string (error message), just post it
            if not result or isinstance(result, str):
                if ctx and msg_id:
                    try:
                        channel = ctx.channel
                        msg_to_edit = await channel.fetch_message(msg_id)
                        await msg_to_edit.edit(content=str(result) or "No updates.")
                    except Exception:
                        pass
                    await ctx.channel.send(str(result))
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if channel:
                        await channel.send(str(result))
                tasks_list.remove((job, ctx, media_type, msg_id, last_count, url_for_content))
                continue

            elif isinstance(result, dict):
                exec_sum = result.get("exec_sum", "")
                notes = result.get("notes", "")
                used_model = result.get("used_model", OPENAI_MODEL)
                custom_prompt_used = result.get("custom_prompt_used", False)

                if ctx and msg_id:
                    try:
                        channel = ctx.channel
                        msg_to_edit = await channel.fetch_message(msg_id)
                        await msg_to_edit.edit(content=f"Processing {media_type} finished!\nCommand processed.")
                    except Exception as e:
                        print(f"[check_tasks] Error editing final message: {e}")

                config = MEDIA_CONFIGS.get(media_type, DEFAULT_MEDIA_CONFIG)
                base_title = config["base_title"]
                disclaimers = config["disclaimers"]
                if OPENAI_MODEL in base_title:
                    base_title = base_title.replace(OPENAI_MODEL, used_model)
                if OPENAI_MODEL in disclaimers:
                    disclaimers = disclaimers.replace(OPENAI_MODEL, used_model)
                if custom_prompt_used:
                    disclaimers = ""

                combined_text_exec = exec_sum + disclaimers
                first_chunks = chunk_text(combined_text_exec, limit=3846)
                notes_pages = chunk_text(notes, limit=3846)
                pages = first_chunks + notes_pages

                report_date = date.today().strftime("%Y-%m-%d")
                unique_id = job.id or "nojobid"
                report_id = f"{media_type}_{report_date}_{unique_id}"
                report_id = report_id[:50]

                view = PersistentPaginatedEmbedView(
                    report_id=report_id,
                    pages=pages,
                    base_title=base_title,
                    current_index=0
                )

                content_tags_list = get_content_tags(exec_sum + "\n" + notes, media_type)
                view.notion_sent = False
                view.notion_tags = content_tags_list
                view.notion_executive = exec_sum
                view.notion_summary = notes
                view.notion_url = url_for_content
                view.notion_media_type = media_type

                first_embed = view._get_embed()

                total_pages = len(pages)
                if custom_prompt_used:
                    if total_pages <= 1:
                        first_embed.description += "\n\nSummary generated, page 1/1."
                    else:
                        first_embed.description += f"\n\nPage 1/{total_pages}, please use the buttons below to navigate between summary pages."

                if ctx:
                    channel = ctx.channel
                    msg = await channel.send(embed=first_embed, view=view)
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if not channel:
                        tasks_list.remove((job, ctx, media_type, msg_id, last_count, url_for_content))
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
                    "channel_id": msg.channel.id,
                    "notion_status": view.notion_status,
                    "notion_executive": view.notion_executive,
                    "notion_summary": view.notion_summary,
                    "notion_tags": view.notion_tags,
                    "notion_url": view.notion_url,
                    "notion_media_type": view.notion_media_type
                }
                r.set(f"report:{report_id}", json.dumps(data))
                bot.add_view(view)

                tasks_list.remove((job, ctx, media_type, msg_id, last_count, url_for_content))

            else:
                # old style (exec_sum, notes) => fallback
                try:
                    exec_sum, notes = result
                except Exception as e:
                    if ctx:
                        await ctx.channel.send(str(result))
                    else:
                        channel = bot.get_channel(DISCORD_CHANNEL_ID)
                        if channel:
                            await channel.send(str(result))
                    tasks_list.remove((job, ctx, media_type, msg_id, last_count, url_for_content))
                    continue

                if ctx and msg_id:
                    try:
                        channel = ctx.channel
                        msg_to_edit = await channel.fetch_message(msg_id)
                        await msg_to_edit.edit(content=f"Processing {media_type} finished!\nCommand processed.")
                    except Exception as e:
                        print(f"[check_tasks] Error editing final message: {e}")

                config = MEDIA_CONFIGS.get(media_type, DEFAULT_MEDIA_CONFIG)
                base_title = config["base_title"]
                disclaimers = config["disclaimers"]
                combined_text_exec = exec_sum + disclaimers

                first_chunks = chunk_text(combined_text_exec, limit=3846)
                notes_pages = chunk_text(notes, limit=3846)
                pages = first_chunks + notes_pages

                report_date = date.today().strftime("%Y-%m-%d")
                unique_id = job.id or "nojobid"
                report_id = f"{media_type}_{report_date}_{unique_id}"
                report_id = report_id[:50]

                view = PersistentPaginatedEmbedView(
                    report_id=report_id,
                    pages=pages,
                    base_title=base_title,
                    current_index=0
                )
                content_tags_list = get_content_tags(exec_sum + "\n" + notes, media_type)
                view.notion_sent = False
                view.notion_tags = content_tags_list
                view.notion_executive = exec_sum
                view.notion_summary = notes
                view.notion_url = url_for_content
                view.notion_media_type = media_type

                first_embed = view._get_embed()
                if ctx:
                    channel = ctx.channel
                    msg = await channel.send(embed=first_embed, view=view)
                else:
                    channel = bot.get_channel(DISCORD_CHANNEL_ID)
                    if not channel:
                        tasks_list.remove((job, ctx, media_type, msg_id, last_count, url_for_content))
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
                    "channel_id": msg.channel.id,
                    "notion_status": view.notion_status,
                    "notion_executive": view.notion_executive,
                    "notion_summary": view.notion_summary,
                    "notion_tags": view.notion_tags,
                    "notion_url": view.notion_url,
                    "notion_media_type": view.notion_media_type,
                }
                r.set(f"report:{report_id}", json.dumps(data))
                bot.add_view(view)

                tasks_list.remove((job, ctx, media_type, msg_id, last_count, url_for_content))

        else:
            # Task still running -> check progress
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
                        if ctx and msg_id:
                            channel = ctx.channel
                            try:
                                msg_to_edit = await channel.fetch_message(msg_id)
                                await msg_to_edit.edit(content=new_content)
                            except Exception as e:
                                print(f"[check_tasks] Error editing message: {e}")
                        idx = tasks_list.index((job, ctx, media_type, msg_id, last_count, url_for_content))
                        tasks_list[idx] = (job, ctx, media_type, msg_id, processed, url_for_content)
            except Exception as e:
                print(f"[check_tasks] Exception checking task progress: {e}")

@tasks.loop(minutes=1)
async def daily_scheduled_tasks():
    now_utc = datetime.now(timezone.utc)
    if now_utc.hour == 10 and now_utc.minute == 0:
        print("[daily_scheduled_tasks] It's 10:00 UTC -> scheduling daily govdigest.")
        job = celery_app.send_task(
            "worker.scrape_governance_forum",
            kwargs={"timeframe": "1d", "only_relevant": True}
        )
        tasks_list.append((job, None, "governance_forum", None, 0, ""))

    if now_utc.hour == 6 and now_utc.minute == 31:
        print("[daily_scheduled_tasks] It's 07:00 UTC -> scheduling daily twitterdigest.")
        job2 = celery_app.send_task(
            "worker.scrape_twitter_digest",
            kwargs={"timeframe": "1d", "only_relevant": True}
        )
        tasks_list.append((job2, None, "twitter_digest", None, 0, ""))

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

# Start loops
check_tasks.start()
check_watchlist_results.start()
# daily_scheduled_tasks is started in on_ready()

bot.run(DISCORD_TOKEN)
