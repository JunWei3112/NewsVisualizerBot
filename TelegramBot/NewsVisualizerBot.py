import config
import telebot
from Databases import CommonDbOperations

# Initialise Telegram bot
bot = telebot.TeleBot(config.BOT_TOKEN, parse_mode=None)

news_articles_cluster = None

def startup_telegram_bot():
    bot.polling(non_stop=True)

@bot.message_handler(commands=['start'])
def start_command(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    visualize_news_button = telebot.types.InlineKeyboardButton(text="Visualize News Article", callback_data="receive_article")
    keyboard.add(visualize_news_button)
    modify_infographic_button = telebot.types.InlineKeyboardButton(text="Modify Infographic", callback_data="modify_infographic")
    keyboard.add(modify_infographic_button)
    greeting_message = 'Greetings! What will you like to do today?'
    bot.send_message(message.chat.id, greeting_message, reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    if call.message:
        if call.data == 'receive_article':
            reply_message = 'What is the article?'
            sent = bot.send_message(call.message.chat.id, reply_message)
            bot.register_next_step_handler(sent, receive_news_article)
        elif call.data == 'modify_infographic':
            reply_message = 'Please send me the infographic that you wish to make changes to.'
            sent = bot.send_message(call.message.chat.id, reply_message)

def receive_news_article(message):
    reply_message = "This is the article that you sent: {}".format(message.text)
    bot.send_message(message.chat.id, reply_message)
    CommonDbOperations.store_news_articles(news_articles_cluster, message.chat.id, message.text)

def startup_database():
    global news_articles_cluster
    news_articles_cluster = CommonDbOperations.startup_database()

if __name__ == '__main__':
    startup_database()
    startup_telegram_bot()
