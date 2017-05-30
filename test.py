# -*- coding: utf-8 -*-
import numpy as np
import sys
import codecs

from konlpy.tag import Twitter
from konlpy.tag import Kkma
konlpy_twitter = Twitter()
kkma = Kkma()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics


def select_test_data(sample_labels, sample_text, i):
	chunksize = len(sample_text)/5
	start = chunksize * i;
	if i == 4:
		end = len(sample_text)
	else:
		end = start + chunksize

	test_labels = sample_labels[start:end]
	test_text = sample_text[start:end]
	train_labels = sample_labels[:start] + sample_labels[end:]
	train_text = sample_text[:start] + sample_text[end:]

	return (test_labels, test_text, train_labels, train_text)


# Read base data.
base_text = []; base_labels = []
"""
for line in codecs.open('./data/base_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(word[0] for word in konlpy_twitter.pos(text, norm=True))
	base_text.append(text)
	base_labels.append(label)

for line in codecs.open('./data/test_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(word[0] for word in konlpy_twitter.pos(text, norm=True))
	base_text.append(text)
	base_labels.append(label)

"""
for line in codecs.open('./data/ex_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	print(text)
	print(konlpy_twitter.pos(text, norm=True))
	text = ' '.join(word[0] for word in konlpy_twitter.pos(text, norm=True))
	base_text.append(text)
	base_labels.append(label)

#== Make test data. ==#
test_text = []; test_labels = []
for line in codecs.open('./data/ex_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(word[0] for word in konlpy_twitter.pos(text, norm=True))
	test_text.append(text)
	test_labels.append(label)

	
train_labels = base_labels	
train_text = base_text

count_vect = CountVectorizer()
train_text_feat = count_vect.fit_transform(train_text)

#for word in count_vect.vocabulary_:
#	print('%s => %s'%(word, count_vect.vocabulary_[word]))
#print(train_text_feat)

clf = MultinomialNB().fit(train_text_feat, train_labels)

test_text = [' '.join(word[0] for word in konlpy_twitter.pos(_, norm=True)) for _ in test_text]

test_text_feat = count_vect.transform(test_text)

for word in count_vect.vocabulary_:
	print('%s => %s'%(word, count_vect.vocabulary_[word]))
print(test_text_feat)

predicted = clf.predict(test_text_feat)
i = 0; correct = 0.0
for _ in test_text:
	if predicted[i] == test_labels[i]:
		print('[CORRECT] ')
		correct = correct + 1
	else:
		print('[INCORRECT] ')
	print('    * Predicted: ' + predicted[i] + ',  * Expected: ' + test_labels[i])
	print('  ' + _)
	i = i+1;
acc = correct / i
print('\n  -> Accuracy: %.5f\n' % (acc))

predicted = clf.predict_proba(test_text_feat)
print(predicted)

clf.classes_
