import tweepy
import time
import csv
from tqdm import tqdm
import argparse

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)


def get_tweets(search_terms, count=5, lang="en"):

    tweets = []
    # Go through each search term:
    # see: https://docs.tweepy.org/en/latest/api.html#search-tweets for params

    for search_term in tqdm(search_terms):
        # the api has a rate limit, so if you hit the limit, you have to wait 15 minutes to continue
        try:
            # surround the terms with double quotes to search exactly
            search_results = api.search("{} -filter:retweets".format(search_term),
                                        lang=lang,
                                        count=count,
                                        tweet_mode="extended"  # get full text
                                        )
            tweets.extend(search_results)
        except tweepy.RateLimitError:
            print("Rate limit reached at {}: waiting 15 min".format(time.ctime()))
            time.sleep(15 * 60)
    return tweets

def write_tweets(tweets, outfile):
    print("writing {0} tweets to {1}".format(len(tweets), outfile))
    print()
    with open(outfile, "w") as outfile:
        writer = csv.writer(outfile)
        intro_line = ["tweet_id", "author_name", "created_at", "text"]
        writer.writerow(intro_line)
        for tweet in tweets:
            line = [tweet.id,
                    tweet.author.name,
                    str(tweet.created_at),
                    tweet.full_text]
            writer.writerow(line)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_terms", nargs="+", default=["good", "bad"])
    parser.add_argument("--outfile", default="tweets.csv")
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--lang", default="en")

    consumer_key = ""
    consumer_secret = ""
    access_token = ""
    access_token_secret = ""

    auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
    auth.set_access_token(access_token, access_token_secret)

    api = tweepy.API(auth)

    args = parser.parse_args()
    tweets = get_tweets(args.search_terms, args.count, args.lang)
    write_tweets(tweets,  args.outfile)
