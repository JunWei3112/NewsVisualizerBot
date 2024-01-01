import praw
import pandas as pd
import json
import config

def convert_to_csv(posts):
    df = pd.DataFrame(posts)
    df.to_csv('DatasetGenerator/posts.csv', sep=',', index=False)

def convert_posts_to_json(posts):
    json_string = json.dumps(posts)
    json_file = open("DatasetGenerator/posts.json", "w")
    json_file.write(json_string)
    json_file.close()

def convert_json_to_posts():
    file_obj = open("DatasetGenerator/posts.json", "r")
    json_content = file_obj.read()
    return json.loads(json_content)

def convert_ids_to_json(ids):
    json_string = json.dumps(ids)
    json_file = open("DatasetGenerator/posts_ids.json", "w")
    json_file.write(json_string)
    json_file.close()

def convert_json_to_ids():
    file_obj = open("DatasetGenerator/posts_ids.json", "r")
    json_content = file_obj.read()
    return json.loads(json_content)

if __name__ == '__main__':

    reddit = praw.Reddit(
        client_id=config.REDDIT_CLIENT_ID,
        client_secret=config.REDDIT_CLIENT_SECRET,
        user_agent='macos:llm-finetune:v1 (by u/AoePhare)'
    )

    subreddit = reddit.subreddit('PhotoshopRequest')
    limit = 1000
    posts = convert_json_to_posts()
    print(f'Number of existing posts: {len(posts)}')
    posts_ids = convert_json_to_ids()
    existing_ids = list()
    for existing_id_json in posts_ids:
        existing_ids.append(existing_id_json['id'])
    # serious_flair = 'Serious :snoo:'

    for submission in subreddit.top(limit=None, time_filter="month"):
        # post_flair = submission.link_flair_text
        # if post_flair == serious_flair and submission.id not in existing_ids:
        if submission.id not in existing_ids:
            submission_id = submission.id
            post_id = {
                "id": submission_id
            }
            posts_ids.append(post_id)
            submission_title = submission.title
            post = {
                "id": submission_id,
                "Title": submission_title
            }
            posts.append(post)

    print(f'Updated number of posts: {len(posts)}')
    convert_to_csv(posts)
    convert_posts_to_json(posts)
    convert_ids_to_json(posts_ids)
