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
	chunk_size = len(sample_text) / 5
	start = chunk_size * i;
	if i == 4:
		end = len(sample_text)
	else:
		end = start + chunk_size

	test_labels = sample_labels[start:end]
	test_text = sample_text[start:end]
	train_labels = sample_labels[:start] + sample_labels[end:]
	train_text = sample_text[:start] + sample_text[end:]
	return (test_labels, test_text, train_labels, train_text)

# 5-fold cross validation
def five_fold_cross_validation(vec_name, vectorizer, kind, kernel, 
		base_labels, base_text, sample_labels, sample_text, f):
	if kind == 0:
		exp_name = "%s" % (vec_name)
	elif kind == 1:
		exp_name = "%s - %s" % (vec_name, kernel)

	f.write(' * ' + exp_name + ':	')
		
	total_acc = 0.0
	for i in range(0, 5):
		test_labels, test_text, _labels, _text = select_test_data(sample_labels, sample_text, i)	
		
		train_labels = base_labels + _labels
		train_text = base_text + _text		
		#train_labels = _labels
		#train_text = _text
		#train_labels = base_labels
		#train_text = base_text

		test_text = [' '.join(word[0] for word in konlpy_twitter.pos(text, norm=True)) for text in test_text]
			
		trained_vectorizer = copy.deepcopy(vectorizer)
		train_text_feat = trained_vectorizer.fit_transform(train_text)
		test_text_feat = trained_vectorizer.transform(test_text)

		if kind == 0:
			trained_clf = MultinomialNB().fit(train_text_feat, train_labels)
		elif kind == 1:
			trained_clf = svm.SVC(kernel=kernel).fit(train_text_feat, train_labels)
			trained_clf.fit(train_text_feat, train_labels)
		
		if kind == 0:
			print(trained_clf.classes_)
			predicted_prob = trained_clf.predict_proba(test_text_feat)
			print(predicted_prob)
			#print('\n  -> Accuracy: %.5f\n' % (acc))
	
		predicted = trained_clf.predict(test_text_feat)
		predicted[0] = 'hi'
		
		j = 0; correct = 0.0
		for label in test_labels:
			if predicted[j] == label:
				#print('[CORRECT] ')
				#if not 'joy' in predicted[j]:
					#print('[CORRECT]  Predict: ' + predicted[j] + ',	Answer: ' + test_labels[j] +'  ' + _)
				correct = correct + 1
			#else:
				#if not 'joy' in predicted[j]:
					#print('[INCORRECT]Predict: ' + predicted[j] + ',	Answer: ' + test_labels[j] +'  ' + _)
			j = j + 1
		acc = correct / j
		total_acc += acc

	f.write('Accuracy avg: %.3f\n' % ((total_acc / 5) * 100))
	return total_acc / 5


# Read base data.
base_text = []; base_labels = []
for line in codecs.open('./data/base_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(word[0] for word in konlpy_twitter.pos(text, norm=True))
	base_text.append(text)
	base_labels.append(label)

# Read sample emotion data for train and test.
sample_text = []; sample_labels = []
for line in codecs.open('./data/test_data.tsv', 'r', 'utf-8'):
	label, text = line.strip().split('\t')
	text = ' '.join(word[0] for word in konlpy_twitter.pos(text, norm=True))

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

# 5-fold cross validation for each classifier and write result file.
f = open('result.txt', 'w')
# NBC
f.write('[NBC Classifier]\n')
for vec_name, vectorizer in vectorizers:
	five_fold_cross_validation(vec_name, vectorizer, 0, '-', 
			base_labels, base_text, sample_labels, sample_text, f)
# SVM
f.write('\n[SVM Classifier]\n')
for vec_name, vectorizer in vectorizers:
	for kernel in kernels:
		five_fold_cross_validation(vec_name, vectorizer, 1, kernel, 
				base_labels, base_text, sample_labels, sample_text, f)

f.close()
