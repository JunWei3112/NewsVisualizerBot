import config
import telebot
from Databases import CommonDbOperations
import ResponseHandler

# Initialise Telegram bot
bot = telebot.TeleBot(config.BOT_TOKEN, parse_mode=None)

database = None

# To keep track of the infographic links that the users have entered
latest_infographic_links = {}

def startup_telegram_bot():
    bot.polling(non_stop=True)

@bot.message_handler(commands=['start'])
def start_command(message):
    keyboard = telebot.types.InlineKeyboardMarkup()
    visualize_news_button = telebot.types.InlineKeyboardButton(text="Visualize News Article", callback_data="receive_article")
    keyboard.add(visualize_news_button)
    modify_infographic_button = telebot.types.InlineKeyboardButton(text="Modify Infographic", callback_data="modify_infographic")
    keyboard.add(modify_infographic_button)
    view_infographic_stats_button = telebot.types.InlineKeyboardButton(text="View Infographic Stats", callback_data="view_infographic_stats")
    keyboard.add(view_infographic_stats_button)
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
            reply_message = 'Please send me the link to the infographic that you wish to modify.'
            sent = bot.send_message(call.message.chat.id, reply_message)
            bot.register_next_step_handler(sent, receive_infographic)
        elif call.data == 'view_infographic_stats':
            reply_message = 'Please send me the link to the infographic that you wish to view the stats for.'
            sent = bot.send_message(call.message.chat.id, reply_message)
            bot.register_next_step_handler(sent, view_infographic_stats)
        elif call.data == 'vote_for_reliability':
            keyboard = telebot.types.InlineKeyboardMarkup()
            vote_reliable_button = telebot.types.InlineKeyboardButton(text="Reliable",
                                                                      callback_data="vote_reliable")
            keyboard.add(vote_reliable_button)
            vote_unreliable_button = telebot.types.InlineKeyboardButton(text="Unreliable",
                                                                        callback_data="vote_unreliable")
            keyboard.add(vote_unreliable_button)

            reply_message = "Do you view this news article as reliable or unreliable?"
            bot.send_message(call.message.chat.id, reply_message, reply_markup=keyboard)
        elif call.data == 'custom_modification':
            reply_message = "What change do you wish to make to this infographic?"
            sent = bot.send_message(call.message.chat.id, reply_message)
            bot.register_next_step_handler(sent, receive_infographic_changes)
        elif call.data == 'vote_reliable':
            link_to_infographic = latest_infographic_links.get(call.message.chat.id)
            if link_to_infographic is None:
                print('User has not entered an infographic link.')
            CommonDbOperations.increase_reliable_vote_count(database, link_to_infographic)
            reply_message = "You have voted this article as reliable!"
            bot.send_message(call.message.chat.id, reply_message)
        elif call.data == 'vote_unreliable':
            link_to_infographic = latest_infographic_links.get(call.message.chat.id)
            if link_to_infographic is None:
                print('User has not entered an infographic link.')
            CommonDbOperations.increase_unreliable_vote_count(database, link_to_infographic)
            reply_message = "You have voted this article as unreliable!"
            bot.send_message(call.message.chat.id, reply_message)

def receive_news_article(message):
    link_to_infographic = CommonDbOperations.receive_news_article(database, message.chat.id, message.text)
    reply_message = "This is the link to access the generated infographic: {}".format(link_to_infographic)
    bot.send_message(message.chat.id, reply_message)

def receive_infographic(message):
    link_to_infographic = message.text
    is_valid_link = CommonDbOperations.check_if_infographic_link_exists(database, link_to_infographic)
    if is_valid_link:
        latest_infographic_links[message.chat.id] = link_to_infographic

        CommonDbOperations.increase_share_count(database, link_to_infographic)

        keyboard = telebot.types.InlineKeyboardMarkup()
        reliability_vote_button = telebot.types.InlineKeyboardButton(text="Vote For Reliability Of News Article",
                                                                     callback_data="vote_for_reliability")
        keyboard.add(reliability_vote_button)
        custom_modification_button = telebot.types.InlineKeyboardButton(text="Type Out Modification",
                                                                        callback_data="custom_modification")
        keyboard.add(custom_modification_button)

        reply_message = "How will you like to modify the infographic?"
        bot.send_message(message.chat.id, reply_message, reply_markup=keyboard)
    else:
        reply_message = "You have sent an invalid link."
        bot.send_message(message.chat.id, reply_message)

def receive_infographic_changes(message):
    user_infographic = latest_infographic_links.get(message.chat.id)
    if user_infographic is None:
        print('User has not entered an infographic link.')
    proposed_user_changes = message.text
    processing_message = 'Generating instructions... ' + '\U000023F3'
    bot.send_message(message.chat.id, processing_message)
    ResponseHandler.generate_intermediate_representation(proposed_user_changes)
    bot.send_message(message.chat.id, "Completed")

def view_infographic_stats(message):
    link_to_infographic = message.text
    is_valid_link = CommonDbOperations.check_if_infographic_link_exists(database, link_to_infographic)
    if is_valid_link:
        CommonDbOperations.increase_share_count(database, link_to_infographic)
        infographic_document = CommonDbOperations.get_infographic_document(database, link_to_infographic)
        infographic_stats = format_infographic_stats(infographic_document)
        bot.send_message(message.chat.id, infographic_stats)
    else:
        reply_message = "You have sent an invalid link."
        bot.send_message(message.chat.id, reply_message)

def format_infographic_stats(infographic_document):
    overview = f"Share Count: {infographic_document['share_count']}\n"
    overview += f"Vote Counts for Reliable: {infographic_document['reliable_vote_count']}\n"
    overview += f"Vote Counts for Unreliable: {infographic_document['unreliable_vote_count']}\n"
    return overview

def startup_database():
    global database
    database = CommonDbOperations.startup_database()

if __name__ == '__main__':
    startup_database()
    startup_telegram_bot()
