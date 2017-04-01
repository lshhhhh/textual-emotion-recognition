# -*- coding: utf-8 -*-

import numpy as np
import sys
import codecs

from konlpy.tag import Twitter
konlpy_twitter = Twitter()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

train_text = []; train_labels = []
for line in codecs.open('emotion_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(konlpy_twitter.morphs(text))
#	print('%s : %s'%(label, text))
	train_text.append(text)
	train_labels.append(label)

count_vect = CountVectorizer()
train_text_feat = count_vect.fit_transform(train_text)

#for word in count_vect.vocabulary_:
#	print('%s => %s'%(word, count_vect.vocabulary_[word]))

#print(train_text_feat)

clf = MultinomialNB().fit(train_text_feat, train_labels)

test_text = [u'오늘 감기에 걸렸다..', u'아싸 오랜만에 놀이동산이다!']
test_text = [' '.join(konlpy_twitter.morphs(_)) for _ in test_text]
for _ in test_text:
	print(_)

test_text_feat = count_vect.transform(test_text)
print(test_text_feat)

predicted = clf.predict(test_text_feat)
print(predicted)

predicted = clf.predict_proba(test_text_feat)
print(predicted)

clf.classes_
