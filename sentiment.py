#Program to analyze Tweets to see if they are pro or anti [THING]
#
# Copyright (C) 2015 Amanda Clark
# Author: Amanda Clark <mandypandy22@gmail.com>

from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from twython import Twython
import configparser
import nltk
from nltk.probability import *
from sklearn.naive_bayes import MultinomialNB


def get_tweets(user):
    """
    A function that uses the Twitter API to download a user's Tweets.
    It retreives the maximum number of Tweets, subject to Twitter API restrictions.
    Reads the keys from the twit_key.ini file, and stores the tweets in a file called
    [username] + .txt. Need to find the URL that had this sample code in it!

    """
    config = configparser.ConfigParser()
    config.read('twit_key.ini')
    CONSUMER_KEY = config['KEYS']['CONSUMER_KEY']
    CONSUMER_SECRET = config['KEYS']['CONSUMER_SECRET']
    ACCESS_KEY = config['KEYS']['ACCESS_KEY']
    ACCESS_SECRET = config['KEYS']['ACCESS_SECRET']
    twitter = Twython(CONSUMER_KEY,CONSUMER_SECRET,ACCESS_KEY,ACCESS_SECRET)
    user_timeline = twitter.get_user_timeline(screen_name=user,count=1)
    lis = user_timeline[0]['id']
    tweets = open(user+".txt", "w")
    lis = [lis]
    for i in range(0, 16):
        user_timeline = twitter.get_user_timeline(screen_name=user, count=200,
                                                  include_retweets=True, max_id=lis[-1])
        for tweet in user_timeline:
            print(tweet['text'])
            lis.append(tweet['id'])
            tweets.write(tweet['text'])

    load_tweets(user+'.txt')


def classify(set_of_words):
    """
    The set of common words is passed, and here is where the classifier classifies the training set, and
    where the set of common words is used to predict user status. Should probably be divided up into at least one
    more function. I think it does too much. See
    http://scikit-learn.org/stable/tutorial/text_analytics/working_with_text_data.html#loading-the-20-newsgroups-dataset

    """
    categories = ['Anti', 'Pro']
    tweets = load_files('/Users/amandaclark/PycharmProjects/SentimentTwitter/Tweets', encoding= 'latin-1',
                        categories=categories, load_content=True)
    count_vect = CountVectorizer()

    X_train_count = count_vect.fit_transform(tweets.data)

    tfidf_transformer = TfidfTransformer()

    X_train_tfidf = tfidf_transformer.fit_transform(X_train_count)

    clf = MultinomialNB().fit(X_train_tfidf, tweets.target)

    docs_new = set_of_words

    X_new_counts = count_vect.transform(docs_new)

    X_new_tfidf = tfidf_transformer.transform(X_new_counts)

    predicted = clf.predict(X_new_tfidf)
    results = dict()
    for doc, category in zip(docs_new, predicted):
        print('%r => %s' % (doc, tweets.target_names[category]))
        results[doc] = tweets.target_names[category]

    numOfAnti = 0
    numOfPro = 0

    for result in results:
        if results[result] == "Anti":
            numOfAnti +=1
        else:
            numOfPro +=1

    if numOfAnti > numOfPro:
        print("The Twitter user is anti [THING]")
    if numOfPro > numOfAnti:
        print("The Twitter user is pro [THING]")
    if numOfAnti == numOfPro:
        print("Cannot determine if user is pro or anti [THING]")


def load_tweets(tweet_file):
    """
    Loads tweets into a form suitable for analysis. Probably could be combined with get_prob.
    See http://www.nltk.org/book/ch01.html for more reference.
    """
    user_tweets = open(tweet_file, 'r')
    text = user_tweets.read()
    text = text.split()
    twts = nltk.Text(text)
    get_prob(twts)


def get_prob(twts):
    """
    Uses the FreqDist from the NLTK toolkit, and creates a set of the most common words in the
    user's Tweet history. Can filter by length, frequency, or any other part of a string.
    TODO: Decide if want to submit to NLTK a fix so that filtering out by (for example) Twitter handle isn't
    handled manually. See http://www.nltk.org/book/ch01.html

    """
    fdist = FreqDist(twts)
    v = sorted(w for w in set(twts) if len(w) > 7 and fdist[w] > 7 and "http" not in w and "//" not in w)
    print(v)
    classify(v)


if __name__ == "__main__":
    """
    Should this be a stand alone program or a separate module?

    """

    user = input("Enter Twitter User: ")
    if user[0] == '@':
        user = user[1:]
        get_tweets(user)
    else:
        get_tweets(user)


