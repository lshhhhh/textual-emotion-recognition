# -*- coding: utf-8 -*-

import tweepy
import os

"""
consumer_key = 'MY CONSUMER KEY'
consumer_secret = 'MY CONSUMER SECRET'
access_token = 'MY ACCESS TOKEN'
access_token_secret = 'MY ACCESS TOKEN SECRET'
"""

# Request for certification.
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)

# Request for access token.
auth.set_access_token(access_token, access_token_secret)

# Create twitter API.
api = tweepy.API(auth)


# Search by keyword.
keyword = u"봄";
search = []

cnt = 1
while (cnt <= 10):
	tweets = api.search(keyword)
	for tweet in tweets:
		search.append(tweet)
	cnt += 1
#print(search[0])

wfile1 = open(os.getcwd() + "/crawling_data_keyword.tsv", mode='w')
data = {}
for tweet in search:
	data['text'] = tweet.text
	wfile1.write(data['text'].encode('utf-8') + '\n')
wfile1.close()


# Get data from timeline.
wfile2 = open(os.getcwd() + "/crawling_data_timeline.tsv", mode='w')
data = {}
for tweet in tweepy.Cursor(api.home_timeline).items(10):
	data['text'] = tweet.text
	wfile2.write(data['text'].encode('utf-8') + '\n')
wfile2.close()

"""
for tweet in tweepy.Cursor(api.friends).items():
for tweet in tweepy.Cursor(api.user_timeline).items():
"""
