import pymongo
import config

# Initialise MongoDB cluster
mongodb_connection_url = config.DB_CONNECTION_STRING.format(
    config.DB_ADMIN_USERNAME,
    config.DB_ADMIN_PW
)
mongodb_client = pymongo.MongoClient(mongodb_connection_url)

def startup_database():
    print("Initializing News Articles Database...")
    database = mongodb_client[config.DB_DATABASE_NAME]
    news_articles_cluster = database[config.DB_ARTICLES_COLLECTION_NAME]
    return news_articles_cluster

def store_news_articles(cluster, chat_id, news_article):
    dict = {"chat_id": chat_id, "article": news_article, "status": "Not Generated"}
    cluster.insert_one(dict)
