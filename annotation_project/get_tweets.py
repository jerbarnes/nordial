import tweepy
import time
import csv

def limit_handled(cursor):
    while True:
        try:
            yield cursor.next()
        except tweepy.RateLimitError:
            time.sleep(15 * 60)


consumer_key = ""
consumer_secret = ""
access_token = ""
access_token_secret = ""

consumer_key = "4Dw2XEDmM530VaYKc9aDkeRb1"
consumer_secret = "ZYcoDaF0wyM73JMluh2jA97CJoZMyyk9cV3GGRbQ3ConvnYuFQ"
access_token = "3331242015-m3vWPmJefEjP8AEuRUqsSJWM1PhhoG1Q1apxdR5"
access_token_secret = "SaPUocF4PUDZEDb38gWwnyPKYlQDiKGJsVbdYeo1p41AH"

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)

api = tweepy.API(auth)


dialekt_terms = ["æ", "eg", "ikk", "itj", "ittje", "itte", "itt"]
tweets = []

# Go through each search term
for term in dialekt_terms:
    # the api has a rate limit of 2 requests per minute, so if you hit the limit, you have to wait 15 minutes to continue
    try:
        search_results = api.search(q=term, count=100, lang="no")
        for result in search_results:
            # If the tweet has at least some content, add to tweets.
            # I chose 10 "tokens" as a simple heuristic
            if len(result.text.split()) > 10:
                tweets.append(result)
    except tweepy.RateLimitError:
        time.sleep(15 * 60)

with open("raw_data.csv", "w") as outfile:
    writer = csv.writer(outfile)
    intro_line = ["tweet_id", "author_name", "created_at", "text"]
    writer.writerow(intro_line)
    for tweet in tweets:
        line = [tweet.id, tweet.author.name, str(tweet.created_at), tweet.text]
        writer.writerow(line)
