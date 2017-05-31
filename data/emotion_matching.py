# -*- coding: utf-8 -*-

import numpy as np
import sys
import codecs

from konlpy.tag import Twitter
konlpy_twitter = Twitter()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


# Read base data.
train_text = []; train_labels = []
for line in codecs.open('./base_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(konlpy_twitter.morphs(text))
	train_text.append(text)
	train_labels.append(label)

for line in codecs.open('./test_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(konlpy_twitter.morphs(text))
	train_text.append(text)
	train_labels.append(label)


# Read sample emotion data for test.
origin_text = []
test_text = []; test_labels = []
for line in codecs.open('./crawling_data.tsv', 'r', 'utf-8'):
	origin_text.append(line)
	text = ' '.join(konlpy_twitter.morphs(line))
	test_text.append(text)


# Make emotion matching.
count_vect = CountVectorizer()
train_text_feat = count_vect.fit_transform(train_text)
clf = MultinomialNB().fit(train_text_feat, train_labels)

test_text = [' '.join(konlpy_twitter.morphs(_)) for _ in test_text]
test_text_feat = count_vect.transform(test_text)

test_labels = clf.predict(test_text_feat)


# Make emotion matching file.
f = codecs.open('./test_data.tsv', 'a', 'utf-8')
file_contents = [];
for i in range(0, len(test_labels)):
	file_contents.append(test_labels[i] + '\t' + origin_text[i])

f.writelines(file_contents)
f.close()

