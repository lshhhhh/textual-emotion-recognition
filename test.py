# -*- coding: utf-8 -*-

import numpy as np
import sys
import codecs

from konlpy.tag import Twitter
konlpy_twitter = Twitter()

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


#== Read base data. ==#
base_text = []; base_labels = []
for line in codecs.open('./data/base_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(konlpy_twitter.morphs(text))
	base_text.append(text)
	base_labels.append(label)

#== Read sample emotion data for train and test. ==#
sample_text = []; sample_labels = []
for line in codecs.open('./data/emotion_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(konlpy_twitter.morphs(text))
	#print('%s : %s'%(label, text))
	sample_text.append(text)
	sample_labels.append(label)

#== 5-fold cross validation. ==#
total_acc = 0.0;
for i in range(0, 5):
	print('\n===== TEST #%d =====\n' % (i+1))

	#== Select test data from sample. ==#
	test_labels, test_text, _labels, _text = select_test_data(sample_labels, sample_text, i)	
	
	train_labels = base_labels + _labels
	train_text = base_text + _text		
	
	count_vect = CountVectorizer()
	train_text_feat = count_vect.fit_transform(train_text)

	#for word in count_vect.vocabulary_:
		#print('%s => %s'%(word, count_vect.vocabulary_[word]))
	#print(train_text_feat)

	clf = MultinomialNB().fit(train_text_feat, train_labels)

	test_text = [' '.join(konlpy_twitter.morphs(_)) for _ in test_text]


	test_text_feat = count_vect.transform(test_text)
	#print(test_text_feat)

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
	total_acc = total_acc + acc
	print('\n  -> Accuracy: %.5f\n' % (acc))

	predicted = clf.predict_proba(test_text_feat)
	print(predicted)

	clf.classes_

print('\n===== TEST END =====')
print('[ACCURACY AVERAGE] %.5f\n' % (total_acc/5))

