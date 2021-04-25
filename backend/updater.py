import sys
import praw
import time
from datetime import datetime
import requests
import threading
import json

redditClient = None

def fetchComments(subreddit, companies, tickers):
    sr_obj = redditClient.subreddit(subreddit)
    companies.extend(tickers)
    try:
        for comment in sr_obj.stream.comments(skip_existing=True):
            for company in companies:
                if company in comment.body:
                    print(comment)
    except Exception as error:
        print("Error {0} occurred while streaming comments from subreddit {1}".format(error))


if __name__=='__main__':
    creds = json.loads(open("creds.json","r").read())
    print(creds)
    redditClient = praw.Reddit(client_id=creds['client_id'],
                               client_secret=creds['client_secret'],
                               password=creds['password'],
                               user_agent=creds['user_agent'],
                               username=creds['username'])

    subreddits = [sr.strip() for sr in open("subreddits","r").read().split(',')]
    companies = [cmp.strip() for cmp in open("companies","r").read().split(',')]
    tickers = [tick.strip() for tick in open("tickers","r").read().split(',')]

    # start fetch thread for every subreddit
    fetch_threads = []
    for sr in subreddits:
        th = threading.Thread(name='fetch_comments_{0}'.format(sr), target=fetchComments, args=(sr, companies, tickers))
        th.start()
        fetch_threads.append(th)

    try:
        while True:
            time.sleep(2)
    except KeyboardInterrupt:
        for th in fetch_threads:
            th.join()


"""

This module is responsible for

Streaming comments 

Stream comments from reddit and write to specified source (stdout or kafka)

"""
