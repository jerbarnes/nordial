import tweepy
import time
import csv
from tqdm import tqdm

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

bokmal = ['jeg har', 'de går', 'jeg skal', 'jeg blir', 'de skal', 'jeg er', 'de blir', 'de har', 'de er', 'dere går', 'dere skal', 'dere blir', 'dere har', 'dere er', 'hun går', 'hun skal', 'hun blir', 'hun har', 'hun er', 'jeg går']

nynorsk = ['eg har', 'dei går', 'eg skal', 'eg blir', 'dei skal', 'eg er', 'dei blir', 'dei har', 'dei er', 'de går', 'dykk går','de skal','dykk skal','de blir','dykk blir','de har','dykk har','de er','dykk er', 'ho gaar', 'ho skal', 'ho blir', 'ho har', 'ho er', 'eg går']

dialect = ['e ha', 'æ ha', 'æ har', 'e har', 'jæ ha', 'eg har', 'eg ha', 'je ha', 'jæ har', 'di går', 'demm går', 'dem går', 'dæmm går', 'dæm går', 'dæi går', 'demm gå', 'dem gå', 'di går', 'domm gå', 'dom gå', 'dømm går', 'døm går', 'dæmm gå', 'dæm gå', 'e ska', 'æ ska', 'jæ ska', 'eg ska', 'je ska', 'i ska', 'ei ska', 'jæi ska', 'je skæ', 'e bli', 'æ bli', 'jæ bli', 'e bi', 'æ blir', 'æ bi', 'je bli', 'e blir', 'i bli', 'di ska', 'dæmm ska', 'dæm ska', 'dæi ska', 'demm ska', 'dem ska', 'domm ska', 'dom ska', 'dømm ska', 'døm ska', 'dæ ska', 'domm ska', 'dom ska', 'æmm ska', 'æm ska', 'eg e', 'æ e', 'e e', 'jæ æ', 'e æ', 'jæ ær', 'je æ', 'i e', 'æg e', 'di bi', 'di bli', 'dæi bli', 'dæmm bli', 'dæm bli', 'di blir', 'demm bli', 'dem bli', 'dæmm bi', 'dæm bi', 'dømm bli', 'døm bli', 'dømm bi', 'døm bi', 'di har', 'di ha', 'dæmm ha', 'dæm ha', 'dæmm har', 'dæm har', 'dæi he', 'demm har', 'dem har', 'demm ha', 'dem ha', 'dæi ha', 'di he', 'dæmm e', 'dæm e', 'di e', 'dæi e', 'demm e', 'dem e', 'di æ', 'dømm æ', 'døm æ', 'demm æ', 'dem æ', 'dei e', 'dæi æ', 'dåkk går', 'dåkke går', 'dåkke gå', 'de går', 'dåkk ska', 'dere ska', 'dåkker ska', 'dåkke ska', 'di ska', 'de ska', 'åkk ska', 'røkk ska', 'døkker ska', 'døkk bli', 'dåkker bi', 'dåkke bli', 'dåkker har', 'dåkker ha', 'dere ha', 'dåkk ha', 'de har', 'dåkk har', 'dere har', 'de ha', 'døkk ha', 'dåkker e', 'dåkk e', 'dåkke e', 'di e', 'dere ær', 'dåkk æ', 'de e', 'økk e', 'døkk æ', 'ho går', 'hu går', 'ho jenng', 'ho gjenng', 'u går', 'o går', 'ho jænng', 'ho gjænng', 'ho jenngg', 'ho gjenngg', 'ho jennge', 'ho gjennge', 'ho gå', 'ho ska', 'hu ska', 'a ska', 'u ska', 'o ska', 'hu skar', 'honn ska', 'ho sjka', 'hænne ska', 'ho bli', 'ho bi', 'o bli', 'ho blir', 'hu bli', 'hu bler', 'hu bi', 'ho bir', 'a blir', 'ho ha', 'ho har', 'ho he', 'hu har', 'hu ha', 'hu he', 'o har', 'o ha', 'hu e', 'ho e', 'hu e', 'ho æ', 'hu æ', 'o e', 'hu ær', 'u e', 'ho ær', 'ho er', 'e går', 'æ går', 'eg går', 'jæ gå', 'jæ går', 'æ gå', 'jæi går', 'e gå']


#dialekt_terms = ["æ", "eg", "ikk", "itj", "ittje", "itte", "itt"]


# Go through each search term
for term_list, outname in [(bokmal, "bokmal.csv"), (nynorsk, "nynorsk.csv"), (dialect, "dialect.csv")]:
    print("Searching for terms for {}".format(outname[:-4]))

    tweets = []
    for term in tqdm(term_list):
        # the api has a rate limit of 2 requests per minute, so if you hit the limit, you have to wait 15 minutes to continue
        try:
            # surround the terms with double quotes to search exactly
            search_query = '\"{}\" -filter:retweets'.format(term)
            search_results = api.search(q=search_query, count=1000, lang="no")
            for result in search_results:
                # If the tweet has at least some content, add to tweets.
                # I chose 10 "tokens" as a simple heuristic
                if len(result.text.split()) > 10:
                    tweets.append(result)
        except tweepy.RateLimitError:
            print("Rate limit reached at {}: waiting 15 min".format(time.ctime()))
            time.sleep(15 * 60)


    # TODO:
    ##############################
    # 1) remove or avoid getting retweets, and other noise
    # 2) remove duplicates
    # 3) get more tweets per search, possibly increase search terms for dialects
    # 4) expand search based on people who often tweet in dialect

    # write to file
    print("writing {0} tweets to {1}".format(len(tweets), outname))
    print()
    with open(outname, "w") as outfile:
        writer = csv.writer(outfile)
        intro_line = ["tweet_id", "author_name", "created_at", "text"]
        writer.writerow(intro_line)
        for tweet in tweets:
            line = [tweet.id, tweet.author.name, str(tweet.created_at), tweet.text]
            writer.writerow(line)
