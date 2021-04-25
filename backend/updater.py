import sys
import praw
import time
import threading
import json

redditClient = None

class CommentsFetcher (threading.Thread):
    die = False
    sr_obj = None
    tickers = []
    companies = []
    def __init__(self, subreddit, companies, tickers, exit_on_fail=False):
        threading.Thread.__init__(self)
        self.name = 'fetch_comments_{0}'.format(subreddit)
        self.companies = companies
        self.tickers = tickers
        self.exit_on_fail = exit_on_fail
        lock = threading.RLock()
        with lock:
            self.sr_obj = redditClient.subreddit(subreddit)

    def run(self):
        while not self.die:
            try:
                self.fetchComments()
            except Exception as e:
                if self.exit_on_fail:
                    raise
                else:
                    print("Thread {1}, Error {0} occurred while streaming comments, continuing".format(e, self.name))

    def join(self):
        self.die = True
        super().join()

    def fetchComments(self):
        search_strings = self.companies + self.tickers
        for comment in self.sr_obj.stream.comments(skip_existing=True, pause_after=5):
            for company in companies:
                if company in comment.body:
                    print(comment.body)


if __name__=='__main__':
    creds = json.loads(open("creds.json","r").read())
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
        th = CommentsFetcher(sr, companies, tickers)
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
