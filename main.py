# Discord imports
import discord
import json
import time
from discord import app_commands
from discord import embeds

# Hugging face imports
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

tokenizer = AutoTokenizer.from_pretrained("openchat/openchat_3.5", use_fast=False)
model = AutoModelForCausalLM.from_pretrained("openchat/openchat_3.5")

# Discord pre-config!
#class aclient(discord.Client):
#
#  def __init__(self):
#    intents = discord.Intents.default()
#    intents.message_content = True
#    super().__init__(intents=discord.Intents.default())
#    self.synced = False  #we use this so the bot doesn't sync commands more than once
#
#  async def on_ready(self):
#    await self.wait_until_ready()
#    if not self.synced:  #check if slash commands have been synced
#      await tree.sync()
#      self.synced = True
#      print(f"We have logged in as {self.user}.")
#
#client = aclient()
#tree = app_commands.CommandTree(client)

# Discord pre-config with clyde fix!
intents = discord.Intents.default()
intents.message_content = True

client = discord.Client(intents=intents)
tree = app_commands.CommandTree(client)


# Testing stuff.
@tree.command(name='test', description='Testing stuff!')
async def test(interaction: discord.Interaction):
  embed = discord.Embed()
  embed.title = "Testing!"
  embed.description = "I am working! I was made with Discord.py!"
  embed.color = discord.Color.green()
  await interaction.response.send_message(embed=embed, ephemeral=True)

# Set channel for NellyAi to use!
@tree.command(name='channel', description='Set channel for NellyAi to use!')
@app_commands.describe(channel = "channel")
async def setchannel(interaction: discord.Interaction, channel: str):

  channel_id = channel

  data = {"channel_id": channel_id}

  # Save channel ID to JSON file
  with open("channel_id.json", "w") as file:
      json.dump(data, file)

  await interaction.response.send_message("Channel ID saved successfully!", ephemeral=True)

# Generate!
@client.event
async def on_message(message):
    print(message.content)
    with open("channel_id.json") as file:
        channel_ids = json.load(file)

    if message.author.bot:
      return

    if message.content.startswith("//"):
      return

    if str(message.channel.id) in channel_ids["channel_id"]:
      await message.channel.typing()

      usrmsg = message.content

      author_mention = message.author.mention

      # Get user input
      user_input = f">> User: {usrmsg}"

      # Encode the new user input and append the eos_token_id
      new_user_input_ids = tokenizer.encode(user_input + tokenizer.eos_token, return_tensors='pt')

      # Generate a response
      chat_history_ids = model.generate(new_user_input_ids, max_length=1000, pad_token_id=tokenizer.eos_token_id)

      # Decode the chat history
      output = tokenizer.decode(chat_history_ids[:, new_user_input_ids.shape[-1]:][0], skip_special_tokens=True)

      output2 = "{}".format(output)

      await message.reply(f"{output2}")

client.run("replace this with your token")
