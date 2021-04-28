import argparse
import json
import math
import praw
import threading
import time

from kafka import KafkaProducer

redditClient = None

class CommentsFetcher (threading.Thread):
    die = False
    sr_obj = None
    companies = {}
    def __init__(self, subreddit, companies, exit_on_fail=False, producer=None, topic=None):
        threading.Thread.__init__(self)
        self.name = 'fetch_comments_{0}'.format(subreddit)
        self.companies = companies
        self.exit_on_fail = exit_on_fail
        self.producer = producer
        self.topic = topic
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
        for comment in self.sr_obj.stream.comments(skip_existing=True, pause_after=5):
            comment_text = comment.body.casefold()
            for ticker in self.companies:
                casefolded_company = self.companies[ticker].casefold()
                if ('{0} '.format(ticker) in comment.body or
                        ' {0}'.format(ticker) in comment.body or
                        '{0} '.format(casefolded_company) in comment_text or
                        ' {0}'.format(casefolded_company) in comment_text):
                    comment_obj = { "ticker": ticker, "text": comment.body, "timestamp": math.ceil(time.time_ns()/1000000) }
                    self.output(comment_obj)
                    break

    def output(self, comment):
        if self.producer is None:
            print(comment)
        else:
            if self.topic is None:
                raise ValueError("topic not supplied")
            key = "{0}_{1}".format(comment["ticker"],comment["timestamp"])
            try:
                key_bytes = bytes(key, encoding='utf-8')
                value = json.dumps(comment_obj)
                value_bytes = bytes(value, encoding='utf-8')
                self.producer.send(self.topic, key=key_bytes, value=value_bytes)
            except Exception as e:
                print("Error {0} occurred while publishing message with key {1}".format(e, key))

if __name__=='__main__':
    parser = argparse.ArgumentParser(description='Stream reddit comments to stdout or kafka topic')
    parser.add_argument('-t', '--topic', metavar='<topic_name>', help='Kafka topic name')
    parser.add_argument('-H', '--host', metavar='<hostname_port>', default='localhost:9092', help='Hostname:port of bootstrap server')
    args = parser.parse_args()
    creds = json.loads(open("creds.json","r").read())
    redditClient = praw.Reddit(client_id=creds['client_id'],
                               client_secret=creds['client_secret'],
                               password=creds['password'],
                               user_agent=creds['user_agent'],
                               username=creds['username'])


    subreddits = [sr.strip() for sr in open("subreddits","r").read().split(',')]
    companies = json.loads(open("companies.json","r").read())

    producer = None
    if args.topic is not None:
       producer = KafkaProducer(bootstrap_servers=[args.host], api_version=(0, 10))

    # start fetch thread for every subreddit
    fetch_threads = []
    for sr in subreddits:
        th = CommentsFetcher(sr, companies, producer, args.topic)
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
