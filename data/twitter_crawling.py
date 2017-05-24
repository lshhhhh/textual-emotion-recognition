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
keywords = [u"우와", u"기분", u"내일", u"너무", u"휴"]
#[u"싫", u"대박", u"오", u"악", u"짱"]
#[u"사랑" ,u"행복", u"우울", u"깜짝", u"헐"]
#[u"ㅠㅠ", u"으으", u"제발", u"진짜", u"좋"]
search = []

for keyword in keywords:
	cnt = 1
	while (cnt <= 10):
		tweets = api.search(keyword)
		for tweet in tweets:
			search.append(tweet)
		cnt += 1

wfile1 = open(os.getcwd() + "/crawling_data.tsv", mode='a')
data = {}
data_list = []
for tweet in search:
	data['text'] = tweet.text
	data_list.append(data['text'].encode('utf-8'))

data_set = set(data_list)
data_list = list(data_set)

for elem in data_list:
	wfile1.write(elem + '\n')
wfile1.close()

"""
# Get data from timeline.
wfile2 = open(os.getcwd() + "/crawling_data_timeline.tsv", mode='w')
data = {}
for tweet in tweepy.Cursor(api.home_timeline).items(10):
	data['text'] = tweet.text
	wfile2.write(data['text'].encode('utf-8') + '\n')
wfile2.close()


for tweet in tweepy.Cursor(api.friends).items():
for tweet in tweepy.Cursor(api.user_timeline).items():
"""
