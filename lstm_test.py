# -*- coding: utf-8 -*-
import numpy as np
import sys
import codecs

from konlpy.tag import Twitter
konlpy_twitter = Twitter()

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.layers import Embedding
from keras.layers import LSTM
from keras.preprocessing import sequence #EDIT


# Select test data from sample data for 5-fold cross validation.
def select_test_data(sample_labels, sample_text, i):
	chunk_size = len(sample_text) / 5
	start = int(chunk_size * i);
	if i == 4:
		end = len(sample_text)
	else:
		end = int(start + chunk_size)

	test_labels = sample_labels[start:end]
	test_text = sample_text[start:end]
	train_labels = sample_labels[:start] + sample_labels[end:]
	train_text = sample_text[:start] + sample_text[end:]
	return (test_labels, test_text, train_labels, train_text)

# Make (text, index) dictionary.
def make_text_index_dic(_text):
	word_set = set()
	for text in _text:
		for word in text:
			word_set.add(word)

	word_dic = {'UNK_WORD':0}#Edit
	i = 1 #EDIT
	for word in word_set:
		word_dic.update({word:i})
		i = i + 1
	return word_dic

# Map text list to index list.
def map_text_to_index(_text, _dic, max_rnn_len):
	x_index = []
	x_element = []
	for text in _text:
		for word in text:
			if word in _dic:	#Edit
				x_element.append(_dic.get(word))	#Edit
			else:
				x_element.append(_dic.get('UNK_WORD'))	#Edit
		#np.array(x_element, dtype=np.int)
		x_index.append(x_element)
		x_element = []
	x_index = sequence.pad_sequences(x_index, maxlen=max_rnn_len) #EDIT
	return x_index

# Map label list to index list.
def map_label_to_index(_labels):
	#label_dic = {'joy':0, 'love':1, 'sadness':2, 'surprise':3, 'anger':4, 'fear':5, 'neutral':6}
	label_dic = {#EDIT
		'joy'     :[1,0,0,0,0,0,0],
		'love'    :[0,1,0,0,0,0,0],
		'sadness' :[0,0,1,0,0,0,0],
		'surprise':[0,0,0,1,0,0,0],
		'anger'   :[0,0,0,0,1,0,0],
		'fear'    :[0,0,0,0,0,1,0],
		'neutral' :[0,0,0,0,0,0,1]}
	y_index = []
	for label in _labels:
		y_index.append(label_dic[label])
	return np.array(y_index, dtype=np.int)

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

# 5-fold cross validation.
max_features = 128
total_acc = 0.0
MAX_RNN_LEN = 50 #EDIT
for i in range(0, 5):
	print('\n===== TEST #%d =====\n' % (i+1))
	test_labels, test_text, _labels, _text = select_test_data(sample_labels, sample_text, i)	
	train_labels = base_labels + _labels
	train_text = base_text + _text

	text_index_dic = make_text_index_dic(train_text + test_text)
	x_train = map_text_to_index(train_text, text_index_dic, MAX_RNN_LEN) #EDIT
	y_train = map_label_to_index(train_labels)
	x_test = map_text_to_index(test_text, text_index_dic, MAX_RNN_LEN) #EDIT
	y_test = map_label_to_index(test_labels)
	
	model = Sequential()
	model.add(Embedding(len(text_index_dic), output_dim=128, input_length=MAX_RNN_LEN, mask_zero=True)) #EDIT
	model.add(LSTM(128))
	#model.add(Dropout(0.5))
	model.add(Dense(7)) #EDIT
	model.add(Activation('softmax')) #EDIT
	model.compile(loss='categorical_crossentropy', 
			optimizer='adam', #EDIT
			metrics=['accuracy'])
            
	print('Fitting...')
	model.fit(x_train, y_train, batch_size=32, epochs=5) #EDIT
	print('Done')

	# Edit (following)
	#score, acc = model.evaluate(x_test, y_test, batch_size=32)
	y_pred = model.predict(x_test)
	y_pred = [np.argmax(_) for _ in y_pred]
	y_true = [np.argmax(_) for _ in y_test]
	print('Prediction =',y_pred)
	print('True =', y_true)	
	accuracy = np.average([y_pred[i] == y_true[i] for i in range(len(y_pred))])
	print('Accuracy =', accuracy)
	
	# print('Score: ', score)
	# print('Accuracy: ', acc)
	# print(model.metrics_names)
	total_acc += accuracy

total_acc /= 5
print('Total Accuracy: ', total_acc)	
