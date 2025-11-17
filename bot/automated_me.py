import asyncio
import csv
import json
import logging
import os
import random
from collections import defaultdict
from pathlib import Path

import discord
import torch
from discord.ext import commands
from dotenv import load_dotenv
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

'''
Discord Bot
'''

load_dotenv()
TOKEN = os.environ["BOT_TOKEN"]
CHANNEL_ID = os.environ["CHANNEL_ID"]
USER_NAME = os.environ["USER_NAME"]

MESSAGE_CSV = "messages.csv"
HISTORY_CSV = "history.csv"
DATA_FILE = Path("projects.json")

try:
    if DATA_FILE.exists():
        with open(DATA_FILE, "r", encoding="utf-8") as f:
            upcoming_projects = json.load(f)
except json.JSONDecodeError:
    upcoming_projects = []

intents = discord.Intents.default()
intents.message_content = True
intents.messages = True
handler = logging.FileHandler(filename='discord.log', encoding='utf-8', mode='w')

bot = commands.Bot(command_prefix="!", intents=intents)

'''
Auto-Me
'''
print("Loading Auto-Me...")

BASE_MODEL = "meta-llama/Meta-Llama-3-8B-Instruct"
TRAINED_MODEL = os.environ["TRAINED_MODEL"]

RESPONSE_PROBABILITY = .5
MESSAGE_DELAY = 0.8

conversation_history = defaultdict(list)
MAX_HISTORY = 10

pending_messages = []
current_batch_key = None
batch_timer = None

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
    bnb_4bit_use_double_quant=True,
)

tokenizer = AutoTokenizer.from_pretrained(TRAINED_MODEL)

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL,
    quantization_config=bnb_config,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    attn_implementation="flash_attention_2",
)

model.resize_token_embeddings(len(tokenizer))
model = PeftModel.from_pretrained(model, TRAINED_MODEL)

print("Auto-Me loaded!")

async def process_and_respond(channel_id):
    """Processes batched messages and responds"""
    global pending_messages, current_batch_key, batch_timer
    
    if not pending_messages:
        current_batch_key = None
        batch_timer = None
        return
    
    combined_message = " ".join(pending_messages)
    channel = bot.get_channel(channel_id)

    pending_messages = []
    current_batch_key = None
    batch_timer = None
    
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(
        None, 
        generate_response, 
        channel_id, 
        combined_message
    )
    
    messages = response.split('<NEWMSG>')
    for i, msg in enumerate(messages):
        msg = msg.strip()
        if msg:
            await channel.send(msg)

async def delayed_response(channel_id, wait_time):
    """Waits and then process the batched messages"""
    await asyncio.sleep(wait_time)
    await process_and_respond(channel_id)

def generate_response(channel_id, user_message):
    conversation_history[channel_id].append({"role": "user", "content": user_message})
    
    if len(conversation_history[channel_id]) > MAX_HISTORY * 2:
        conversation_history[channel_id] = conversation_history[channel_id][-MAX_HISTORY*2:]
    
    messages = [
        {"role": "system", "content": f"You are {USER_NAME}, and you respond exactly in his natural Discord style and tone."}
    ]
    messages.extend(conversation_history[channel_id][-MAX_HISTORY:])
    
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to(model.device)
    
    outputs = model.generate(
        **inputs,
        max_new_tokens=70,
        temperature=0.9,
        top_p=0.9,
        do_sample=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    
    response = tokenizer.decode(outputs[0][inputs['input_ids'].shape[1]:], skip_special_tokens=True).strip()

    conversation_history[channel_id].append({"role": "assistant", "content": response})
    
    return response

@bot.command(name='reset')
async def reset_conversation(ctx):
    """Reset the auto-me bot to factory settings. Should fix any quirkiness"""
    global pending_messages, current_batch_key, batch_timer

    channel_id = ctx.channel.id

    conversation_history[channel_id] = []

    if current_batch_key and current_batch_key[0] == channel_id and batch_timer:
        batch_timer.cancel()
        batch_timer = None
        current_batch_key = None
        pending_messages = []
    
    await ctx.send("Conversation history reset and all pending responses cleared!")

@bot.command(name='probability')
async def set_probability(ctx, prob: float):
    """Sets response probability. To turn off, set to 0"""
    global RESPONSE_PROBABILITY
    if 0 <= prob <= 1:
        RESPONSE_PROBABILITY = prob
        await ctx.send(f"Response probability set to {prob*100:.1f}%")
    else:
        await ctx.send("Probability must be between 0 and 1")


'''
Logging
'''

def log_message(curr_csv, message, status: str):
    """Append a message to the CSV file with a clear status column."""
    with open(curr_csv, "a", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            message.id,
            message.author.id,
            message.author.name,
            message.author.display_name,
            message.channel.id,
            message.created_at.isoformat(),
            len(message.content),
            message.attachments != [],
            status,
            message.content
        ])

def log_message_jsonl(message, status: str):
    with open("messages.jsonl", "a", encoding="utf-8") as f:
        json_obj = {
            "id": message.id,
            "author_id": message.author.id,
            "author_name": message.author.name,
            "display_name": message.author.display_name,
            "channel_id": message.channel.id,
            "created_at": message.created_at.isoformat(),
            "length": len(message.content),
            "has_attachment": len(message.attachments) > 0,
            "status": status,
            "text": message.content
        }
        f.write(json.dumps(json_obj, ensure_ascii=False) + "\n")

@bot.event
async def on_ready():
    print(f'We have logged in as {bot.user}')

@bot.event
async def on_message(message):
    global pending_messages, current_batch_key, batch_timer

    if message.author.bot or message.author == bot.user:
        return
    log_message(MESSAGE_CSV, message, status="new")
    await bot.process_commands(message)

    channel_id = message.channel.id
    user_id = message.author.id
    key = (channel_id, user_id)
    
    if current_batch_key == key:
        pending_messages.append(message.content)
        return
    
    # To maintain coherency, the bot ignores all users except the one it is
    # focusing on.
    if current_batch_key is not None:
        return
    
    if random.random() >= RESPONSE_PROBABILITY:
        return
    
    current_batch_key = key
    pending_messages.append(message.content)
    
    wait_time = random.uniform(5, 15)
    batch_timer = asyncio.create_task(
        delayed_response(channel_id, user_id, wait_time)
    )


@bot.event
async def on_message_edit(_, after):
    if after.author.bot:
        return
    log_message(MESSAGE_CSV, after, status="edited")

@bot.event
async def on_message_delete(message):
    if message.author.bot:
        return
    log_message(MESSAGE_CSV, message, status="deleted")

'''
User Commands
'''

def choose_new(chosen=set()) -> str | None:
    folder = "bot_imgs"
    files = os.listdir(folder)
    if not files:
        return

    counter = 0
    while True:
        chosen_image = random.choice(files)
        if chosen_image not in chosen:
            break
        counter += 1
        if counter == 100:
            return

    chosen.add(chosen_image)
    path = os.path.join(folder, chosen_image)
    return path

bot.img_spam_counts = {}

@bot.command()
async def image(ctx):
    if ctx.author.bot:
        return

    user_id = ctx.author.id

    if user_id != OWNER_ID:
        if not bot.img_spam_counts.get(user_id, 0):
            bot.img_spam_counts[user_id] = 1
        elif bot.img_spam_counts.get(user_id, 0) < 3:
            bot.img_spam_counts[user_id] = bot.img_spam_counts[user_id] + 1
        else:
            await ctx.send("You've had enough for today.")
            return


    path = choose_new()
    if not path:
        await ctx.send("Seems we've run out of images.")
        return

    await ctx.send(file=discord.File(path))

def add_project():
    with open(DATA_FILE, "w", encoding="utf-8") as f:
        json.dump(upcoming_projects, f, ensure_ascii=False, indent=2)

@bot.command()
async def in_progress(ctx):
    if ctx.author.bot:
        return
    
    upcoming = 0
    if (len(upcoming_projects[1:])):
        upcoming = len(upcoming_projects[1:])
    
    await ctx.send(f"In Progress: {upcoming_projects[0]}")
    await ctx.send(f"Upcoming: {upcoming}")


'''
Admin Commands
'''

OWNER_ID = os.environ["OWNER_ID"]

def is_owner(ctx):
    return ctx.author.id == OWNER_ID

@bot.command()
@commands.check(is_owner)
async def queue_project(ctx, *, project):
    upcoming_projects.append(project)
    add_project()
    await ctx.send(f"Queued: {project}")

@bot.command()
@commands.check(is_owner)
async def quit(ctx):
    print(f'{bot.user} shutting down...')
    await ctx.send("Shutting down...")
    await bot.close()

@bot.command()
@commands.check(is_owner)
async def wipecsv(ctx):
    try:
        open("messages.csv", "w", encoding="utf-8").close()
        await ctx.send("CSV logs have been wiped.")
    except Exception as e:
        await ctx.send(f"Error wiping CSV: {e}. Requesting manual removal.")

'''
Scraping
'''

@bot.command()
@commands.check(is_owner)
async def scrape_all(ctx):
    '''
    Scrape every message in the channel provided in the .env
    '''
    channel = ctx.guild.get_channel(CHANNEL_ID)
    if channel is None:
        await ctx.send("Invalid channel ID. I messed up.")
        return

    await ctx.send(f"Beginning full scrape of {channel.mention}...")

    total = 0
    last_id = None

    while True:
        history = []
        async for message in channel.history(
            limit=100,
            before=discord.Object(id=last_id) if last_id else None,
            oldest_first=False
        ):
            history.append(message)

        if not history:
            break

        for message in history:
            if message.author.bot:
                continue
            log_message_jsonl(message, status="scraped_full")
            total += 1

        last_id = history[-1].id

        await asyncio.sleep(1.0)

    await ctx.send(f"Full scrape complete. Logged {total} messages.")

bot.run(TOKEN, log_handler=handler, log_level=logging.WARNING)
