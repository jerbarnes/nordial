import tweepy
import time
import csv
import stanza
import re
import numpy as np
from tqdm import tqdm
from nltk import FreqDist, ngrams
from nltk.corpus import stopwords

bokmal = ['jeg har', 'de går', 'jeg skal', 'jeg blir', 'de skal', 'jeg er', 'de blir', 'de har', 'de er', 'dere går', 'dere skal', 'dere blir', 'dere har', 'dere er', 'hun går', 'hun skal', 'hun blir', 'hun har', 'hun er', 'jeg går']

nynorsk = ['eg har', 'dei går', 'eg skal', 'eg blir', 'dei skal', 'eg er', 'dei blir', 'dei har', 'dei er', 'de går', 'dykk går','de skal','dykk skal','de blir','dykk blir','de har','dykk har','de er','dykk er', 'ho gaar', 'ho skal', 'ho blir', 'ho har', 'ho er', 'eg går']

dialect = ['e ha', 'æ ha', 'æ har', 'e har', 'jæ ha', 'eg har', 'eg ha', 'je ha', 'jæ har', 'di går', 'demm går', 'dem går', 'dæmm går', 'dæm går', 'dæi går', 'demm gå', 'dem gå', 'di går', 'domm gå', 'dom gå', 'dømm går', 'døm går', 'dæmm gå', 'dæm gå', 'e ska', 'æ ska', 'jæ ska', 'eg ska', 'je ska', 'i ska', 'ei ska', 'jæi ska', 'je skæ', 'e bli', 'æ bli', 'jæ bli', 'e bi', 'æ blir', 'æ bi', 'je bli', 'e blir', 'i bli', 'di ska', 'dæmm ska', 'dæm ska', 'dæi ska', 'demm ska', 'dem ska', 'domm ska', 'dom ska', 'dømm ska', 'døm ska', 'dæ ska', 'domm ska', 'dom ska', 'æmm ska', 'æm ska', 'eg e', 'æ e', 'e e', 'jæ æ', 'e æ', 'jæ ær', 'je æ', 'i e', 'æg e', 'di bi', 'di bli', 'dæi bli', 'dæmm bli', 'dæm bli', 'di blir', 'demm bli', 'dem bli', 'dæmm bi', 'dæm bi', 'dømm bli', 'døm bli', 'dømm bi', 'døm bi', 'di har', 'di ha', 'dæmm ha', 'dæm ha', 'dæmm har', 'dæm har', 'dæi he', 'demm har', 'dem har', 'demm ha', 'dem ha', 'dæi ha', 'di he', 'dæmm e', 'dæm e', 'di e', 'dæi e', 'demm e', 'dem e', 'di æ', 'dømm æ', 'døm æ', 'demm æ', 'dem æ', 'dei e', 'dæi æ', 'dåkk går', 'dåkke går', 'dåkke gå', 'de går', 'dåkk ska', 'dere ska', 'dåkker ska', 'dåkke ska', 'di ska', 'de ska', 'åkk ska', 'røkk ska', 'døkker ska', 'døkk bli', 'dåkker bi', 'dåkke bli', 'dåkker har', 'dåkker ha', 'dere ha', 'dåkk ha', 'de har', 'dåkk har', 'dere har', 'de ha', 'døkk ha', 'dåkker e', 'dåkk e', 'dåkke e', 'di e', 'dere ær', 'dåkk æ', 'de e', 'økk e', 'døkk æ', 'ho går', 'hu går', 'ho jenng', 'ho gjenng', 'u går', 'o går', 'ho jænng', 'ho gjænng', 'ho jenngg', 'ho gjenngg', 'ho jennge', 'ho gjennge', 'ho gå', 'ho ska', 'hu ska', 'a ska', 'u ska', 'o ska', 'hu skar', 'honn ska', 'ho sjka', 'hænne ska', 'ho bli', 'ho bi', 'o bli', 'ho blir', 'hu bli', 'hu bler', 'hu bi', 'ho bir', 'a blir', 'ho ha', 'ho har', 'ho he', 'hu har', 'hu ha', 'hu he', 'o har', 'o ha', 'hu e', 'ho e', 'hu e', 'ho æ', 'hu æ', 'o e', 'hu ær', 'u e', 'ho ær', 'ho er', 'e går', 'æ går', 'eg går', 'jæ gå', 'jæ går', 'æ gå', 'jæi går', 'e gå']


def normalize(tweet):
    #change mentions to @AT
    normalized = re.sub(r"@[\w]+", "@AT", tweet)
    #change urls to URL
    normalized = re.sub(r"https://[\w\.\/]+", "URL", normalized)
    return normalized

def remove_to_exclude(fd, to_exclude):
    new_fd = FreqDist()
    for key, value in fd.items():
        if type(key) is str:
            if key not in to_exclude:
                new_fd[key] = value
        else:
            in_exclude = False
            for k in key:
                if k in to_exclude:
                    in_exclude = True
            if in_exclude is False:
                new_fd[key] = value
    return new_fd


def get_most_common_features(sents, exclude=True):
    nonwords = set(["@at", "url", ",", ".", "at", "@", '…', '"', "'", ":", "!", "?", "-", ")", "(", '...', "/", "\\", "#", '»', '«', ']', '[', ";", '&amp'])
    no_stopwords = set(stopwords.words("norwegian"))
    single_letters = set("a b c d f g h j k l m n p q r s t u v w x y z".split())
    to_exclude = nonwords.union(no_stopwords).union(single_letters)
    tokens = [l.split() for l in sents]
    tokens = [token for sent in tokens for token in sent]
    if exclude:
        uni_fd = remove_to_exclude(FreqDist(tokens), to_exclude)
        bi_fd = remove_to_exclude(FreqDist(ngrams(tokens, 2)), to_exclude)
        tri_fd = remove_to_exclude(FreqDist(ngrams(tokens, 3)), to_exclude)
    else:
        uni_fd = FreqDist(tokens)
        bi_fd = FreqDist(ngrams(tokens, 2))
        tri_fd = FreqDist(ngrams(tokens, 3))
    return uni_fd, bi_fd, tri_fd

def combine_grams(fds):
    fd = FreqDist()
    for f in fds:
        for k, v in f.items():
            fd[k] = v
    return fd

def import_data(csv_file, tokenizer):
    # pull in current tweets and get frequency of uni, bi- and trigrams
    tweets = []
    sents = []
    with open(csv_file) as infile:
        reader = csv.reader(infile)
        next(reader)
        for id, name, ca, tweet in reader:
            tweets.append(tweet)
            normalized = normalize(tweet)
            doc = tokenizer(normalized)
            for sent in doc.sentences:
                tokenized = " ".join([token.text.lower() for token in sent.tokens])
                sents.append(tokenized)
    # remove any possible duplicates
    sents = list(set(sents))
    return sents

def log_likelihood(fd1, fd2):
    fd = FreqDist()
    # calculate expected values following Rayson and Garside (2000) Comparing Corpora using Frequency Profiling
    #
    #
    # Expected values for a term = c*(a+b) / (c + d)
    # Where a and b are the freqency of the term in fd1 and fd2, respectively
    # and c and d are the total counts of terms in fd1 and fd2, respectively
    c = sum(fd1.values())
    d = sum(fd2.values())
    for term in fd1.keys():
        a = fd1[term] + 0.00000000000000001
        b = fd2[term] + 0.00000000000000001
        E1 = c*(a + b) / (c + d) + 0.00000000000000001
        E2 = d*(a + b) / (c + d) + 0.00000000000000001
        #
        log_likelihood = 2 * ((a*np.log2(a/E1)) + (b*np.log2(b/E2)))
        fd[term] = log_likelihood
    return fd

def chi_squared(fd1, fd2, p=0.05):
    fd = FreqDist()
    # calculate expected values following Rayson and Garside (2000) Comparing Corpora using Frequency Profiling
    #
    #
    # Expected values for a term = c*(a+b) / (c + d)
    # Where a and b are the freqency of the term in fd1 and fd2, respectively
    # and c and d are the total counts of terms in fd1 and fd2, respectively
    significance_dict = {0.05: 3.84, 0.01: 6.63, 0.001: 10.83}
    c = sum(fd1.values())
    d = sum(fd2.values())
    for term in fd1.keys():
        a = fd1[term] + 0.00000000000000001
        b = fd2[term] + 0.00000000000000001
        E1 = c*(a + b) / (c + d) + 0.00000000000000001
        E2 = d*(a + b) / (c + d) + 0.00000000000000001
        # X^2 = sum of (observed - expected)**2 / expected
        chi = ((a - E1)**2/ E1) + ((b - E2)**2 / E2)
        # p < 0.05 = 3.84, p < 0.01 = 6.63, p < 0.001 = 10.83
        if chi > significance_dict[p]:
            fd[term] = chi
    return fd

if __name__ == "__main__":

    nn_tokenizer = stanza.Pipeline(lang="nn", processors="tokenize")
    nb_tokenizer = stanza.Pipeline(lang="nb", processors="tokenize")

    dialect_sents = import_data("dialect.csv", nn_tokenizer)
    di_uni, di_bi, di_tri = get_most_common_features(dialect_sents)

    bokmal_sents = import_data("bokmal.csv", nb_tokenizer)
    b_uni, b_bi, b_tri = get_most_common_features(bokmal_sents)

    nynorsk_sents = import_data("nynorsk.csv", nn_tokenizer)
    n_uni, n_bi, n_tri = get_most_common_features(nynorsk_sents)

    # log_likelihood
    uni_f = log_likelihood(di_uni, b_uni)
    bi_f = log_likelihood(di_bi, b_bi)
    tri_f = log_likelihood(di_tri, b_tri)

    ll_terms = combine_grams([uni_f, bi_f, tri_f])
    print("Log Likelihood(dialect, bokmål)")
    for term, metric in ll_terms.most_common(30):
        print("-- {0} - {1:.1f}".format(term, metric))
    print("\n\n")

    # chi squared
    uni_f = chi_squared(di_uni, b_uni)
    bi_f = chi_squared(di_bi, b_bi)
    tri_f = chi_squared(di_tri, b_tri)

    chi_terms = combine_grams([uni_f, bi_f, tri_f])
    print("Chi_squred(dialect, bokmål)")
    for term, metric in chi_terms.most_common(30):
        print("-- {0} - {1:.1f}".format(term, metric))
