import discord
from discord.ext import tasks
from dotenv import load_dotenv
import os
from celery_config import app as celery_app
import redis
import json

bot = discord.Bot()
load_dotenv()

r = redis.Redis(host='localhost', port=6379, db=2)

tasks_list = []


@bot.event
async def on_ready():
    print("Logged in as " + bot.user.name)


@bot.slash_command()
async def list(ctx):
    current_watchlist = [item.decode('utf-8')
                         for item in r.lrange('watchlist', 0, -1)]
    await ctx.respond("Current watchlist: " + ", ".join(current_watchlist))


@bot.slash_command()
async def add(ctx, twitter_url: str):
    r.rpush('watchlist', twitter_url)
    await ctx.respond(f"Added {twitter_url} to watchlist")


@bot.slash_command()
async def remove(ctx, twitter_url: str):
    r.lrem('watchlist', 1, twitter_url)
    await ctx.respond(f"Removed {twitter_url} from watchlist")


@bot.slash_command()
async def process(ctx, space_url: str):
    await ctx.respond(f"Processing space URL {space_url}... I'll notify you when it's done.")
    job = celery_app.send_task('worker.scrape_space', args=[space_url])

    tasks_list.append((job, ctx))


@tasks.loop(seconds=5)  # adjust the interval as needed
async def check_tasks():
    for job, ctx in tasks_list[:]:  # iterate over a copy of the list
        if job.ready():
            # If the task is ready, get the result
            result = job.get()
            print(result)
            # Only send a message and remove the task if the result is not None
            if result is not None:
                await ctx.respond(f"Executive Summary: {result['exec_sum']}")
                tasks_list.remove((job, ctx))


@tasks.loop(seconds=60)
async def check_watchlist_results():
    channel_id = os.getenv("DISCORD_CHANNEL_ID")
    channel = bot.get_channel(channel_id)

    for result_bytes in r.lrange('watchlist_results', 0, -1):
        result_str = result_bytes.decode('utf-8')
        result_dict = json.loads(result_str)

        await channel.send(f"Result: {result_dict['exec_sum']}")

        # Remove the result from the list
        r.lrem('watchlist_results', 1, result_bytes)

check_tasks.start()
check_watchlist_results.start()

bot.run(os.getenv("DISCORD_TOKEN"))
