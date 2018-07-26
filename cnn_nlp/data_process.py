import tensorflow as tf
import numpy as np
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import hangul

test = "안녕하세요 안녕하세영웅 preprocess 중 입니다. 123456 ???? /// !!!! ㅡㅜㅜ "
test1 = ["안녕하세요 안녕 안녕하 안녕하세 안녕하세여", "안녕, 안녕하세여, 안녕하세여"]
test_input = hangul.normalize(test, english=False, number=False, punctuation=False)

def cut(contents, cut=2):
	results = []
	for content in contents:
		words = content.split()
		result = []
		for word in words:
			result.append(word[:cut])
		results.append(' '.join([token for token in result]))
	return results

def make_outputs(points, threshold=2.5):
	results = np.zeros((len(points), 2))
	for idx, point in enumerate(points):
		if point > threshold:
			results[idx,0] = 1
		else:
			results[idx,1] = 0
	return results

def load_data_and_labels_zz(positive_data_file, negative_data_file):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    positive_examples = list(open(positive_data_file, "r", encoding='utf-8').readlines())
    positive_examples = [s.strip() for s in positive_examples]
    negative_examples = list(open(negative_data_file, "r", encoding='utf-8').readlines())
    negative_examples = [s.strip() for s in negative_examples]
    # Split by words
    x_text = positive_examples + negative_examples
    x_text = [sent for sent in x_text]
    # Generate labels
    positive_labels = [[0, 1] for _ in positive_examples]
    # print(positive_labels)
    negative_labels = [[1, 0] for _ in negative_examples]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

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
# with open('intent.json') as f:
# 	data = json.load(f)

# data = data['intents']
# file = open('result.txt', 'a')
# for i in range(len(data)):
# 	print(data[i]['intent'])
# 	file.write(data[i]['intent'])
# 	if 'chitchat' in data[i]['intent']:
# 		print(data[i]['examples'])
# 		for i in range()
# 		file.write(data[i]['examples'])
# file.close()

test = [1.0,5.0,4.0,2.0, 1.0,3.0]
test_one = make_outputs(test)

# for i in range(len(test_one)):
# 	print(test_one[i])
# 	print(test_one[i][1])
print("Loading data...")
name_list = ["외모", "가족"]
x_text , y = load_data_and_labels("", name_list)

# Build Vocabulary
print(x_text)
max_document_length = max([len(x.split(" ")) for x in x_text])
print(max_document_length)
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print(x)
#Randomly shuffle data
np.random.seed(10)
shuffle_indices = np.random.permutation(np.arange(len(y)))
x_shuffled = x[shuffle_indices]
y_shuffled = y[shuffle_indices]



# print(x)
# print(x_shuffled)
# print(y_shuffled)

# test1 = [[1,2,3], [4,5,6]]
# print(test1)
# test = np.array(test1)
# print(test)
# shuffle_indices = np.random.permutation(np.arange(len(test)))
# test_shuffled = test[shuffle_indices]
# print(test_shuffled)