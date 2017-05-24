# -*- coding: utf-8 -*-

import numpy as np
import sys
import codecs
import copy

from konlpy.tag import Twitter
konlpy_twitter = Twitter()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics


# Select test data from sample data for 5-fold cross validation.
def select_test_data(sample_labels, sample_text, i):
	chunksize = len(sample_text) / 5
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
for line in codecs.open('./data/base_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(konlpy_twitter.morphs(text))
	base_text.append(text)
	base_labels.append(label)


# Read sample emotion data for train and test.
sample_text = []; sample_labels = []
for line in codecs.open('./data/test_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(konlpy_twitter.morphs(text))
	#print('%s : %s'%(label, text))
	sample_text.append(text)
	sample_labels.append(label)


# Make vectorizers and kernels.
vectorizers = [
	('freq-1gram', CountVectorizer(ngram_range=(1, 1))),
	('freq-2gram', CountVectorizer(ngram_range=(1, 2))),
	('freq-3gram', CountVectorizer(ngram_range=(1, 3))),
	('tfidf-1gram', TfidfVectorizer(ngram_range=(1, 1))),
	('tfidf-2gram', TfidfVectorizer(ngram_range=(1, 2))),
	('tfidf-3gram', TfidfVectorizer(ngram_range=(1, 3)))
]

kernels = ['linear', 'rbf', 'poly']


f = open('result.txt', 'w')

classifier_num = 0
# 5-fold cross validation for each classifier.
for vec_name, vectorizer in vectorizers:
	for kernel in kernels:
		exp_name = "%s_%s"%(vec_name, kernel)

		classifier_num = classifier_num + 1
		f.write('%d) Classifier: ' % classifier_num + exp_name + '\n')
		
		svm_total_acc = 0.0
		nbc_total_acc = 0.0
		for i in range(0, 10):
			n = i % 5
			#print('===== TEST #%d =====\n'%(n+1))
			test_labels, test_text, _labels, _text = select_test_data(sample_labels, sample_text, n)	
	
			train_labels = base_labels + _labels
			train_text = base_text + _text		
			test_text = [' '.join(konlpy_twitter.morphs(_)) for _ in test_text]
			
			trained_vectorizer = copy.deepcopy(vectorizer)
			train_text_feat = trained_vectorizer.fit_transform(train_text)
			test_text_feat = trained_vectorizer.transform(test_text)

			if i < 5:
				trained_clf = svm.SVC(kernel=kernel).fit(train_text_feat, train_labels)
				trained_clf.fit(train_text_feat, train_labels)
			else:
				trained_clf = MultinomialNB().fit(train_text_feat, train_labels)
		
			predicted = trained_clf.predict(test_text_feat)
	
			j = 0; correct = 0.0
			for _ in test_text:
				if predicted[j] == test_labels[j]:
					#print('[CORRECT] ')
					correct = correct + 1
				#else:
					#print('[INCORRECT] ')
				#print('    * Predicted: ' + predicted[i] + ',  * Expected: ' + test_labels[i])
				#print('  ' + _)
				j = j + 1
			acc = correct / j

			if i < 5:
				svm_total_acc += acc
			else:
				nbc_total_acc += acc
			#print('\n  -> Accuracy: %.5f\n' % (acc))

			#predicted = trained_clf.predict_proba(test_text_feat)
			#print(predicted)

			trained_clf.classes_

		#print('\n===== TEST END =====')
		f.write('  - SVM ACCURACY AVERAGE: %.5f\n' % (svm_total_acc / 5))
		f.write('  - NBC ACCURACY AVERAGE: %.5f\n\n' % (nbc_total_acc / 5))

