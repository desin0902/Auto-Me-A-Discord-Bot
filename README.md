# Auto-Me-A-Discord-Bot
Most of the functions in this bot, as of now, are DIY. I included the necessary database functions, scrapers, logging, and other tools that should make it pretty seamless to make your own bot. A few of the miscellanous functions, like sending images, are mostly for my own use with my friends. It is a personal project, after all. It ended up being a really good way to learn scraping, API usage, and LLM training via Axolotl and pytorch.

Files and Folders:

analytics: A folder that contains everything data analysis and data pruning related. Should be easy enough to use or edit the given functions to get a usable dataset. They're a little utilitarian right now, classes have been keeping me absurdly busy, but they get the job done.

bot: Contains everything necessary to run and connect to the Discord API and use your own trained LLM to chat with your friends. More features to be added soon. Private information, like your bot token, should be stored in an .env file.
