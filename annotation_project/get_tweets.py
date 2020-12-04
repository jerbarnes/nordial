import tweepy
import time

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
