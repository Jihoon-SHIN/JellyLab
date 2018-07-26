# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import re
import tensorflow as tf
import random
import hangul

# Data Loading function
def loading_data(data_path, name_list, eng=True, num=True, punc=False):
	# agony = ["외모","가족","학업","취업","직장생활","진로","친구","이성","이웃","성격"]
	agony = name_list
	contents = []
	labels = []
	count = 0
	for a in agony:
		f = open(data_path+a+"_v7.txt",'r')
		content = f.readlines()
		count += len(content)
		for i in range(0, len(content)):
			content[i] = content[i].split("\n")[0]
			content[i] = hangul.normalize(content[i], english=eng, number=num, punctuation=punc)
			contents.append(content[i])
		for j in range(0,len(content)):
			labels.append(a)
	return contents, labels


# Cut the data 
def cut(contents, cut=2):
	results = []
	for content in contents:
		words = content.split()
		result = []
		for word in words:
			result.append(word[:cut])
		results.append(' '.join([token for token in result]))
	return results


# Divide train/test set function
def devide(x, y, train_prop):
	random.seed(1234)
	x = np.array(x)
	y = np.array(y)
	tmp = np.random.permutation(np.arange(len(x)))
	x_tr = x[tmp][:round(train_prop * len(x))]
	y_tr = y[tmp][:round(train_prop * len(y))]
	x_te = x[tmp][-(len(x)-round(train_prop*len(x)))]
	y_te = y[tmp][-(len(x)-round(train_prop*len(y)))]
	return x_tr, x_te, y_tr, y_te


def load_data_and_labels(data_path="", name_list=[], eng=True, num=True, punc=False):
	x_text =[]
	x_labels = []
	for idx, file_name in enumerate(name_list):
		zeros = [0,0,0,0,0,0,0,0,0,0]
		file = data_path + file_name + '_v7.txt'
		data_examples = list(open(file, "r").readlines())
		data_examples = [s.strip() for s in data_examples]
		data_examples = [hangul.normalize(s, english=eng, number=num, punctuation=punc) for s in data_examples]
		x_text = x_text + data_examples
		zeros[idx] = 1
		for num in range(len(data_examples)):
			x_labels.append(zeros) 
	x_labels = np.array(x_labels)
	return [x_text, x_labels]

def batch_iter(data,batch_size, num_epochs, shuffle = True):
	data = np.array(data)
	data_size = len(data)
	num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
	for epoch in range(num_epochs):
		# Shuffle the data at each epoch
		if shuffle:
			shuffle_indices = np.random.permutation(np.arange(data_size))
			shuffled_data = data[shuffle_indices]
		else:
			shuffled_data = data
		for batch_num in range(num_batches_per_epoch):
			start_index = batch_num * batch_size
			end_index = min((batch_num + 1) * batch_size, data_size)
			yield shuffled_data[start_index:end_index]

name_list = ["외모","가족","학업","취업","직장생활","진로","친구","이성","이웃","성격"]
x_text, x_labels = load_data_and_labels("", name_list)
# print(x_text)
# print(x_labels)
# print(x_labels.shape)

