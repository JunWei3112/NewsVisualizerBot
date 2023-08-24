import config
import telebot

# Initialise Telegram bot
bot = telebot.TeleBot(config.BOT_TOKEN, parse_mode=None)

def start_telegram_bot():
    bot.polling(non_stop=True)

@bot.message_handler(commands=['start'])
def start_command(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    callback_button = telebot.types.InlineKeyboardButton(text="Visualize News Article", callback_data="receive_article")
    keyboard.add(callback_button)
    greeting_message = 'Greetings, {}!'.format(message.chat.id)
    bot.send_message(message.chat.id, greeting_message, reply_markup=keyboard)

@bot.callback_query_handler(func=lambda call: True)
def callback_inline(call):
    if call.message:
        if call.data == 'receive_article':
            reply_message = 'What is the article?'
            sent = bot.send_message(call.message.chat.id, reply_message)
            bot.register_next_step_handler(sent, receive_news_article)

def receive_news_article(message):
    reply_message = "This is the article that you sent: {}".format(message.text)
    bot.send_message(message.chat.id, reply_message)

if __name__ == '__main__':
    start_telegram_bot()

