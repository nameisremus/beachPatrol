import os
import io
import logging
import json
import re
from datetime import date, datetime, timezone
from functools import partial
import redis

from config import TG_BOT_TOKEN, OPENAI_MODELS_LIST, OPENAI_MODEL, BOT_PASSWORD, TG_ECOSYSTEM_UPDATES_GROUPID, REDIS_HOST, REDIS_PORT
from core.utils import chunk_text, PersistentPaginatedMessage
from core.integrations.notion_integration import send_tracked_content
from core.core import format_for_telegram, get_content_tags
from telegram import Update, BotCommand, InputFile
from telegram.constants import ParseMode
from telegram.ext import (
    ApplicationBuilder,
    CommandHandler,
    CallbackQueryHandler,
    ContextTypes,
    MessageHandler,
    filters
)

from tasks.celery_config import app as celery_app
from core.integrations.notion_integration import send_tracked_content
from core.media_configs import (
    MEDIA_CONFIGS,
    DEFAULT_MEDIA_CONFIG,
    get_telegram_media_config,
    get_media_type_from_command,
    get_telegram_media_slug
)

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)

r = redis.Redis(host=REDIS_HOST, port=REDIS_PORT, decode_responses=True)

# dict to store pending Telegram tasks.
telegram_tasks = {}


def parse_arg_pairs(args_list):
    """
    Parses arguments for a command in 'key=value' format (case-insensitive for keys).
    Special handling for 'prompt=': if encountered, consumes all subsequent tokens
    as part of the prompt.
    
    Example usage:
      /command http://some/url parse_comments=True model="gpt-4" prompt="Write a summary"
    
    Returns (main_args, extra_kwargs) where:
    - main_args is a list of any positional args (e.g., the URL)
    - extra_kwargs is a dict of any 'key=value' pairs. Surrounding quotes in 'value'
      are stripped if present.
    """
    main_args = []
    extra_kwargs = {}
    skip_until_end = False

    i = 0
    while i < len(args_list):
        item = args_list[i].strip()
        if skip_until_end:
            i += 1
            continue

        # If it's the first argument and looks like a URL, treat it as main arg.
        if i == 0 and (item.startswith("http://") or item.startswith("https://")):
            main_args.append(item)
            i += 1
            continue

        if "=" in item:
            k, v = item.split("=", 1)
            k = k.strip().lower()
            v = v.strip()

            # Strip surrounding quotes if present
            if (v.startswith('"') and v.endswith('"')) or (v.startswith("'") and v.endswith("'")):
                v = v[1:-1]

            if k == "prompt":
                # prompt=some text, plus all subsequent tokens appended
                prompt_parts = [v]
                j = i + 1
                while j < len(args_list):
                    prompt_parts.append(args_list[j])
                    j += 1
                extra_kwargs["prompt"] = " ".join(prompt_parts)
                skip_until_end = True
                i = j
            else:
                extra_kwargs[k] = v
                i += 1
        else:
            main_args.append(item)
            i += 1

    return main_args, extra_kwargs

def is_chat_authorized(chat_id: int) -> bool:
    return r.sismember("authorized_telegram_chats", str(chat_id))

async def password_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /password <BOT_PASSWORD>
    Used to unlock the bot in this chat.
    """
    if not context.args:
        await update.message.reply_text("Usage: /password <BOT_PASSWORD>")
        return

    supplied_pass = context.args[0]
    if supplied_pass == BOT_PASSWORD:
        r.sadd("authorized_telegram_chats", str(update.effective_chat.id))
        await update.message.reply_text("✅ Correct password. This chat is now unlocked.")
    else:
        await update.message.reply_text("❌ Incorrect password. Please try again.")


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /start command."""
    welcome_text = (
        "Welcome to Beach Patrol Telegram Bot!\n"
        "Available commands:\n"
        "  /ping\n"
        "  /generate_summary\n"
        "  /generate_tweet_summary\n"
        "  /generate_gov_digest\n"
        "  /generate_twitter_digest\n"
        "  /generate_twitter_account_summary\n"
        "  /list_available_models\n\n"
        "To unlock commands, use /password <BOT_PASSWORD>."
    )
    await update.message.reply_text(welcome_text, parse_mode=None)

async def ping(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Handle /ping command."""
    if not is_chat_authorized(update.effective_chat.id):
        await update.message.reply_text("Please unlock the bot first using /password <BOT_PASSWORD>.")
        return

    await update.message.reply_text("Pong! 🏓", parse_mode=None)

async def list_available_models(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /list_available_models
    Lists all models from OPENAI_MODELS_LIST, one per line.
    """
    if not is_chat_authorized(update.effective_chat.id):
        await update.message.reply_text("Please unlock the bot first using /password <BOT_PASSWORD>.")
        return

    if not OPENAI_MODELS_LIST or not any(m.strip() for m in OPENAI_MODELS_LIST):
        await update.message.reply_text(
            "No models available (Check OPENAI_MODELS in .env).",
            parse_mode=None
        )
        return
    msg = "Available Models:\n"
    for m in OPENAI_MODELS_LIST:
        if m.strip():
            msg += f"- {m.strip()}\n"
    await update.message.reply_text(msg, parse_mode=None)

async def generate_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /generate_summary <url> [model=xxx] [prompt="..."]
    
    Determines if the URL is a Twitter Space, YouTube video, Twitter/X status, 
    or fallback to an article. If it's a Twitter status link, we call the tweet-summary
    worker with parse_comments=False.
    """
    if not is_chat_authorized(update.effective_chat.id):
        await update.message.reply_text("Please unlock the bot first using /password <BOT_PASSWORD>.")
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /generate_summary <url> [model=xxx] [prompt=\"...\"]",
            parse_mode=None
        )
        return

    main_args, extra = parse_arg_pairs(context.args)
    if not main_args:
        await update.message.reply_text("Please provide a URL.", parse_mode=None)
        return

    url = main_args[0].strip()
    chosen_model = extra.get("model", "")
    chosen_prompt = extra.get("prompt", "")

    # 1) If Twitter Space
    if "twitter.com/i/spaces" in url or "x.com/i/spaces" in url:
        await update.message.reply_text(f"Processing Twitter Space URL {url} ... please wait", parse_mode=None)
        job = celery_app.send_task(
            "worker.scrape_space",
            args=[url],
            kwargs={"model": chosen_model, "prompt": chosen_prompt}
        )
        command_type = "twitter_space"
    # 2) If YouTube
    elif "youtube.com/watch" in url or "youtu.be" in url:
        await update.message.reply_text(f"Processing YouTube URL {url} ... please wait", parse_mode=None)
        job = celery_app.send_task(
            "worker.scrape_youtube_video",
            args=[url],
            kwargs={"model": chosen_model, "prompt": chosen_prompt}
        )
        command_type = "youtube"
    # 3) If Twitter / X status link
    elif ("twitter.com/" in url or "x.com/" in url) and "/status/" in url:
        await update.message.reply_text(f"Processing tweet summary for {url} ... please wait", parse_mode=None)
        # parse_comments always False in /generate_summary
        job = celery_app.send_task(
            "worker.scrape_tweet_summary",
            kwargs={
                "url": url,
                "parse_comments": False,
                "model": chosen_model,
                "prompt": chosen_prompt
            }
        )
        command_type = "tweet_summary"
    # 4) Otherwise fallback to article
    else:
        await update.message.reply_text(f"Processing article URL {url} ... please wait", parse_mode=None)
        job = celery_app.send_task(
            "worker.scrape_article",
            args=[url],
            kwargs={"model": chosen_model, "prompt": chosen_prompt}
        )
        command_type = "article"

    telegram_tasks[job.id] = {
        "job": job,
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "command_type": command_type,
        "url": url
    }

async def generate_tweet_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /generate_tweet_summary <tweet_url> [parse_comments=True|False] [model=xxx] [prompt="..."]
    
    Processes a tweet or thread. parse_comments=True means it will also fetch replies.
    """
    if not is_chat_authorized(update.effective_chat.id):
        await update.message.reply_text("Please unlock the bot first using /password <BOT_PASSWORD>.")
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /generate_tweet_summary <tweet_url> [parse_comments=True|False] [model=xxx] [prompt=\"...\"]",
            parse_mode=None
        )
        return

    main_args, extra = parse_arg_pairs(context.args)
    if not main_args:
        await update.message.reply_text("Please provide a tweet URL.", parse_mode=None)
        return

    url = main_args[0]
    if "?" in url:
        url = url.split("?", 1)[0]

    parse_comments_str = extra.get("parse_comments", "false")
    parse_comments = parse_comments_str.lower() in ("true", "1", "yes")

    chosen_model = extra.get("model", "")
    chosen_prompt = extra.get("prompt", "")

    await update.message.reply_text(
        f"Processing tweet summary for URL {url} with parse_comments={parse_comments} ... please wait",
        parse_mode=None
    )
    job = celery_app.send_task("worker.scrape_tweet_summary", kwargs={
        "url": url,
        "parse_comments": parse_comments,
        "model": chosen_model,
        "prompt": chosen_prompt
    })
    telegram_tasks[job.id] = {
        "job": job,
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "command_type": "tweet_summary",
        "url": url,
        "parse_comments": parse_comments
    }

async def generate_gov_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /generate_gov_digest [timeframe] [relevancy_filter]
    Processes governance forum topics.
    timeframe: e.g., "1d" (default: "1d")
    relevancy_filter: "true" or "false" (default: true)
    """
    if not is_chat_authorized(update.effective_chat.id):
        await update.message.reply_text("Please unlock the bot first using /password <BOT_PASSWORD>.")
        return

    timeframe = context.args[0] if len(context.args) >= 1 else "1d"
    relevancy_filter = True
    if len(context.args) >= 2:
        val = context.args[1].lower()
        relevancy_filter = val in ("true", "1", "yes")
    await update.message.reply_text(
        f"Processing governance forum updates... timeframe={timeframe}, relevancyFilter={relevancy_filter}",
        parse_mode=None
    )
    job = celery_app.send_task(
        "worker.scrape_governance_forum",
        kwargs={"timeframe": timeframe, "only_relevant": relevancy_filter}
    )
    telegram_tasks[job.id] = {
        "job": job,
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "command_type": "governance_forum"
    }

async def generate_twitter_digest(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /generate_twitter_digest [timeframe] [relevancy_filter]
    Processes a digest for multiple Twitter accounts.
    timeframe: e.g., "1d" or "2d" (default: "1d")
    relevancy_filter: "true" or "false" (default: true)
    """
    if not is_chat_authorized(update.effective_chat.id):
        await update.message.reply_text("Please unlock the bot first using /password <BOT_PASSWORD>.")
        return

    timeframe = context.args[0] if len(context.args) >= 1 else "1d"
    relevancy_filter = True
    if len(context.args) >= 2:
        val = context.args[1].lower()
        relevancy_filter = val in ("true", "1", "yes")
    await update.message.reply_text(
        f"Processing Twitter digest... timeframe={timeframe}, relevancyFilter={relevancy_filter}",
        parse_mode=None
    )
    job = celery_app.send_task(
        "worker.scrape_twitter_digest",
        kwargs={"timeframe": timeframe, "only_relevant": relevancy_filter}
    )
    telegram_tasks[job.id] = {
        "job": job,
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "command_type": "twitter_digest"
    }

async def generate_twitter_account_summary(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    /generate_twitter_account_summary <username> [timeframe] [model=xxx] [prompt="..."]
    Summarizes tweets for a single Twitter user.
    """
    if not is_chat_authorized(update.effective_chat.id):
        await update.message.reply_text("Please unlock the bot first using /password <BOT_PASSWORD>.")
        return

    if not context.args:
        await update.message.reply_text(
            "Usage: /generate_twitter_account_summary <username> [timeframe] [model=xxx] [prompt=\"...\"]",
            parse_mode=None
        )
        return

    main_args, extra = parse_arg_pairs(context.args)
    username = main_args[0]
    timeframe = main_args[1] if len(main_args) >= 2 else "1d"
    chosen_model = extra.get("model", "")
    chosen_prompt = extra.get("prompt", "")

    await update.message.reply_text(
        f"Processing Twitter summary for @{username}, timeframe={timeframe} ... please wait",
        parse_mode=None
    )
    job = celery_app.send_task("worker.scrape_twitter_account_summary", kwargs={
        "username": username,
        "timeframe": timeframe,
        "model": chosen_model,
        "prompt": chosen_prompt
    })
    telegram_tasks[job.id] = {
        "job": job,
        "chat_id": update.effective_chat.id,
        "message_id": update.message.message_id,
        "command_type": "twitter_account",
        "url": None
    }

async def check_pending_tasks(context: ContextTypes.DEFAULT_TYPE):
    """
    This function is scheduled in the Telegram JobQueue.
    Every few seconds it iterates over pending tasks (in telegram_tasks),
    checks if a celery job is ready, and if so, sends the final result to the user.
    If the result spans multiple pages, a PersistentPaginatedMessage is created
    (with the same buttons) even if there's only one page.
    """
    bot = context.bot
    finished_jobs = []
    for job_id, meta in list(telegram_tasks.items()):
        job = meta["job"]
        if job.ready():
            try:
                result = job.get()
            except Exception as e:
                result = f"Error retrieving result: {e}"
            if not result or isinstance(result, str):
                await bot.send_message(
                    chat_id=meta["chat_id"],
                    text=str(result) or "No updates",
                    parse_mode=ParseMode.HTML,
                )
            else:
                exec_sum  = result.get("exec_sum", "")
                notes     = result.get("notes", "")
                used_model = result.get("used_model", OPENAI_MODEL)
                custom_prompt_used = result.get("custom_prompt_used", False)

                command_type = meta.get("command_type", "update")
                display_media_type = get_media_type_from_command(command_type)
                telegram_slug = get_telegram_media_slug(command_type)
                url_for_content = meta.get("url", "")
                parse_comments = meta.get("parse_comments", False)

                if parse_comments is False and custom_prompt_used and notes == exec_sum:
                    notes = ""

                content_tags_list = get_content_tags((exec_sum or "") + "\n" + (notes or ""), display_media_type)

                config = get_telegram_media_config(telegram_slug)
                base_title = config["base_title"]
                disclaimers = config["disclaimers"]

                # Replace default model in disclaimers/title with used_model
                if OPENAI_MODEL in base_title:
                    base_title = base_title.replace(OPENAI_MODEL, used_model)
                if OPENAI_MODEL in disclaimers:
                    disclaimers = disclaimers.replace(OPENAI_MODEL, used_model)

                if custom_prompt_used:
                    disclaimers = ""

                telegram_exec_sum = format_for_telegram(exec_sum or "")
                telegram_notes = format_for_telegram(notes or "")
                telegram_exec_sum = telegram_exec_sum.replace("```html", "").replace("```", "")
                telegram_notes = telegram_notes.replace("```html", "").replace("```", "")

                first_page_text = telegram_exec_sum.strip()
                if disclaimers.strip():
                    first_page_text += "\n\n" + disclaimers.strip()

                first_chunks = chunk_text(first_page_text, limit=3846)
                notes_pages = chunk_text(telegram_notes, limit=3846)
                pages = first_chunks + notes_pages

                report_date = date.today().strftime("%Y-%m-%d")
                unique_id = job.id or "nojobid"
                telegram_report_id = f"{telegram_slug}_{report_date}_{unique_id}"[:40]

                if custom_prompt_used:
                    if len(pages) == 1:
                        pages[0] += "\n\nSummary generated, page 1/1."
                    else:
                        total_pages = len(pages)
                        pages[0] += f"\n\nPage 1/{total_pages}, please use the button below to navigate between summary pages."

                view = PersistentPaginatedMessage(
                    report_id=telegram_report_id,
                    pages=pages,
                    base_title=base_title,
                    current_index=0
                )
                view.notion_sent = False
                view.notion_executive = exec_sum
                view.notion_summary = notes
                view.notion_tags = content_tags_list
                view.notion_url = url_for_content
                view.notion_media_type = display_media_type

                sent_msg = await bot.send_message(
                    chat_id=meta["chat_id"],
                    text=view.get_current_text(),
                    reply_markup=view.build_keyboard(),
                    parse_mode=ParseMode.HTML
                )
                view.message_id = sent_msg.message_id
                view.chat_id = sent_msg.chat.id
                view.save_state()

            finished_jobs.append(job_id)
    for job_id in finished_jobs:
        del telegram_tasks[job_id]

async def persistent_callback_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Handles callback queries for persistent pagination.
    """
    query = update.callback_query
    await query.answer()
    data = query.data.split("|", 2)
    if len(data) < 3:
        return
    action = data[1]
    report_id = data[2]
    state_json = r.get(f"report:{report_id}")
    if not state_json:
        await query.edit_message_text(text="Session expired.", parse_mode=ParseMode.HTML)
        return
    state = json.loads(state_json)
    paginated_msg = PersistentPaginatedMessage.from_dict(report_id, state)
    total_pages = len(paginated_msg.pages)
    if action == "first":
        paginated_msg.current_index = 0
    elif action == "previous":
        if paginated_msg.current_index > 0:
            paginated_msg.current_index -= 1
    elif action == "next":
        if paginated_msg.current_index < total_pages - 1:
            paginated_msg.current_index += 1
    elif action == "last":
        paginated_msg.current_index = len(paginated_msg.pages) - 1
    elif action == "download":
        combined_text = "\n\n----- PAGE BREAK -----\n\n".join(paginated_msg.pages)
        buffer = io.StringIO(combined_text)
        await context.bot.send_document(
            chat_id=paginated_msg.chat_id,
            document=InputFile(buffer, filename=f"{paginated_msg.base_title}.txt")
        )
        return
    elif action == "notion":
        paginated_msg.notion_status = "sending"
        r.set(f"report:{report_id}", json.dumps(paginated_msg.to_dict()))
        try:
            await query.edit_message_reply_markup(reply_markup=paginated_msg.build_keyboard())
        except Exception as e:
            if "Message is not modified" not in str(e):
                raise e

        notion_exec = paginated_msg.notion_executive
        if paginated_msg.notion_media_type in ("Twitter Digest", "Twitter Account Digest"):
            if "ℹ️" in notion_exec:
                notion_exec = notion_exec.split("ℹ️", 1)[0].strip()

        combined_text = "\n\n----- PAGE BREAK -----\n\n".join(paginated_msg.pages)
        try:
            send_tracked_content(
                media_type=paginated_msg.notion_media_type,
                url=paginated_msg.notion_url,
                executive_summary=notion_exec or "No executive summary",
                summary=paginated_msg.notion_summary or "No full summary",
                tags=paginated_msg.notion_tags
            )
            paginated_msg.notion_status = "sent"
            new_keyboard = paginated_msg.build_keyboard()
            try:
                await query.edit_message_reply_markup(reply_markup=new_keyboard)
            except Exception as e:
                if "Message is not modified" not in str(e):
                    raise e
            try:
                await query.edit_message_text(
                    text=paginated_msg.get_current_text(),
                    reply_markup=new_keyboard,
                    parse_mode=ParseMode.HTML
                )
            except Exception as e:
                if "Message is not modified" not in str(e):
                    raise e
        except Exception as e:
            await query.edit_message_text(
                text=f"Failed to send to Notion: {e}",
                parse_mode=ParseMode.HTML
            )
            return
    r.set(f"report:{report_id}", json.dumps(paginated_msg.to_dict()))
    try:
        await query.edit_message_text(
            text=paginated_msg.get_current_text(),
            reply_markup=paginated_msg.build_keyboard(),
            parse_mode=ParseMode.HTML
        )
    except Exception as e:
        if "Message is not modified" in str(e):
            pass
        else:
            raise e

async def set_commands(application):
    """
    Register commands with Telegram so that users see them when they type "/".
    """
    commands = [
        BotCommand("ping", "Check if the bot is online"),
        BotCommand("generate_summary", "Generate summary: /generate_summary <url> [model=] [prompt=]"),
        BotCommand("generate_tweet_summary", "Generate tweet summary: /generate_tweet_summary <tweet_url> [parse_comments=] [model=] [prompt=]"),
        BotCommand("generate_gov_digest", "Generate gov digest: /generate_gov_digest [timeframe] [relevancy_filter]"),
        # BotCommand("generate_twitter_digest", "Generate Twitter digest: /generate_twitter_digest [timeframe] [relevancy_filter]"),
        BotCommand("generate_twitter_account_summary", "Summarize a Twitter user: /generate_twitter_account_summary <username> [timeframe] [model=] [prompt=]"),
        BotCommand("list_available_models", "List all available OpenAI models"),
        BotCommand("password", "Unlock the bot: /password <BOT_PASSWORD>")
    ]
    await application.bot.set_my_commands(commands)

def rehydrate_persistent_views(application):
    """
    On bot startup, scan Redis for keys "report:*" and re-register them.
    """
    keys = r.keys("report:*")
    for key in keys:
        report_id = key.removeprefix("report:")
        state_json = r.get(key)
        if not state_json:
            continue
        state = json.loads(state_json)
        paginated_msg = PersistentPaginatedMessage.from_dict(report_id, state)
        try:
            application.bot.edit_message_text(
                chat_id=paginated_msg.chat_id,
                message_id=paginated_msg.message_id,
                text=paginated_msg.get_current_text(),
                reply_markup=paginated_msg.build_keyboard(),
                parse_mode=ParseMode.HTML
            )
        except Exception as e:
            logger.error(f"Failed to rehydrate report {report_id}: {e}")

# Regex to match a Twitter or X link with /status/
tweet_url_pattern = re.compile(r"(https?://(?:x\.com|twitter\.com)/[^/\s]+/status/\d+[^\s]*)", re.IGNORECASE)

async def handle_plain_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    This handler is meant for group chats and runs on any non-command message.
    If the message text has a Twitter or X link with /status/, automatically
    enqueue a tweet summary. The user sees a quick "Detected tweet" message.
    """
    if not is_chat_authorized(update.effective_chat.id):
        return  # or optionally: prompt them to /password

    message_text = update.message.text or ""
    match = tweet_url_pattern.search(message_text)
    if match:
        tweet_url = match.group(1)
        if "?" in tweet_url:
            tweet_url = tweet_url.split("?", 1)[0]

        await update.message.reply_text(
            f"Detected tweet link, summarizing:\n{tweet_url}\nPlease wait..."
        )
        job = celery_app.send_task("worker.scrape_tweet_summary", kwargs={
            "url": tweet_url,
            "parse_comments": False
        })
        telegram_tasks[job.id] = {
            "job": job,
            "chat_id": update.effective_chat.id,
            "message_id": update.message.message_id,
            "command_type": "tweet_summary",
            "url": tweet_url
        }

async def daily_scheduled_tasks(context: ContextTypes.DEFAULT_TYPE):
    """
    Runs every minute, checking the current UTC time.
    If it's 10:00 UTC, schedule the daily governance forum digest.
    If it's 06:31 UTC, schedule the daily Twitter digest.
    """
    now_utc = datetime.now(timezone.utc)

    # 1) Governance forum at 15:00 UTC
    if now_utc.hour == 15 and now_utc.minute == 0:
        logger.info("[daily_scheduled_tasks] It's 15:00 UTC -> scheduling daily govdigest.")
        job = celery_app.send_task(
            "worker.scrape_governance_forum",
            kwargs={"timeframe": "1d", "only_relevant": True}
        )
        # Store in telegram_tasks so check_pending_tasks picks it up
        telegram_tasks[job.id] = {
            "job": job,
            "chat_id": int(TG_ECOSYSTEM_UPDATES_GROUPID),
            "message_id": None,  # not triggered by user command
            "command_type": "governance_forum"
        }

    # 2) Twitter digest at 13:00 UTC
    if now_utc.hour == 13 and now_utc.minute == 00:
        logger.info("[daily_scheduled_tasks] It's 13:00 UTC -> scheduling daily twitterdigest.")
        job2 = celery_app.send_task(
            "worker.scrape_twitter_digest",
            kwargs={"timeframe": "1d", "only_relevant": True}
        )
        telegram_tasks[job2.id] = {
            "job": job2,
            "chat_id": int(TG_ECOSYSTEM_UPDATES_GROUPID),
            "message_id": None,
            "command_type": "twitter_digest"
        }

def main():
    application = ApplicationBuilder().token(TG_BOT_TOKEN).build()

    application.add_handler(CommandHandler("password", password_command))

    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("ping", ping))
    application.add_handler(CommandHandler("list_available_models", list_available_models))
    application.add_handler(CommandHandler("generate_summary", generate_summary))
    application.add_handler(CommandHandler("generate_tweet_summary", generate_tweet_summary))
    application.add_handler(CommandHandler("generate_gov_digest", generate_gov_digest))
    application.add_handler(CommandHandler("generate_twitter_digest", generate_twitter_digest))
    application.add_handler(CommandHandler("generate_twitter_account_summary", generate_twitter_account_summary))
    application.add_handler(
        MessageHandler(filters.TEXT & ~filters.COMMAND, handle_plain_message)
    )
    application.add_handler(CallbackQueryHandler(persistent_callback_handler, pattern="^persistent\\|"))

    application.job_queue.run_repeating(check_pending_tasks, interval=5, first=5)

    async def register_commands_job(context: ContextTypes.DEFAULT_TYPE):
        await set_commands(application)
    application.job_queue.run_once(register_commands_job, when=0)

     #Add the daily ecosystem governance update task job to run every minute
    application.job_queue.run_repeating(daily_scheduled_tasks, interval=60, first=30)

    rehydrate_persistent_views(application)
    application.run_polling()

if __name__ == '__main__':
    main()
