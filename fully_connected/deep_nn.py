#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tool as tool
import time
import os
from tensorflow.python.client import device_lib

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'


#print(device_lib.list_local_devices())

agony = ["외모","가족","학업","취업","직장생활","진로","친구","이성","이웃","성격"]
contents = []
title = []
result = []
count = 0
for a in agony:
    f = open(a+"_v7.txt",'r')
    content = f.readlines()
    count += len(content)
    for i in range(0, len(content)):
        content[i] = content[i].split("\n")[0]
        contents.append(content[i])
    for j in range(0,len(content)):
    	title.append(a)
    	if a == "외모":
    		result.append([0])
    	elif a == "가족":
    		result.append([1])
    	elif a == "학업":
    		result.append([2])
    	elif a == "취업":
    		result.append([3])
    	elif a == "직장생활":
    		result.append([4])
    	elif a == "진로":
    		result.append([5])
    	elif a == "친구":
    		result.append([6])
    	elif a == "이성":
    		result.append([7])
    	elif a == "이웃":
    		result.append([8])
    	elif a == "성격":
    		result.append([9])

word_to_ix, ix_to_word = tool.make_dict_all_cut(contents, minlength=0, maxlength=3, jamo_delete=True)
encoder_size = 100
decoder_size = 3
encoderinputs, decoderinputs, targets_, targetweights = \
    tool.make_inputs(contents, title, word_to_ix,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)
# encoderinputs & result
mix = []
for i in range(len(encoderinputs)):
	mix.append(encoderinputs[i] + result[i])
def train_test_split(records, testratio=0.2):
    np.random.seed(int(len(records)/2))
    np.random.shuffle(records)
    sizerecords = len(records)
    if sizerecords == 0:
        raise ValueError("At least one data required")
    testsize = int(sizerecords * testratio)
    train = records[:-testsize]
    test = records[-testsize:]
    return train, test
train, test = train_test_split(mix)
train_input = []
train_output = []
test_input = []
test_output = []
for i in range(0,len(train)):
	train_input.append(train[i][:100])
	if train[i][100] == 0:
		train_output.append([1,0,0,0,0,0,0,0,0,0])
	elif train[i][100] == 1:
		train_output.append([0,1,0,0,0,0,0,0,0,0])
	elif train[i][100] == 2:
		train_output.append([0,0,1,0,0,0,0,0,0,0])
	elif train[i][100] == 3:
		train_output.append([0,0,0,1,0,0,0,0,0,0])
	elif train[i][100] == 4:
		train_output.append([0,0,0,0,1,0,0,0,0,0])
	elif train[i][100] == 5:
		train_output.append([0,0,0,0,0,1,0,0,0,0])
	elif train[i][100] == 6:
		train_output.append([0,0,0,0,0,0,1,0,0,0])
	elif train[i][100] == 7:
		train_output.append([0,0,0,0,0,0,0,1,0,0])
	elif train[i][100] == 8:
		train_output.append([0,0,0,0,0,0,0,0,1,0])
	elif train[i][100] == 9:
		train_output.append([0,0,0,0,0,0,0,0,0,1])
for i in range(0,len(test)):
	test_input.append(test[i][:100])
	if test[i][100] == 0:
		test_output.append([1,0,0,0,0,0,0,0,0,0])
	elif test[i][100] == 1:
		test_output.append([0,1,0,0,0,0,0,0,0,0])
	elif test[i][100] == 2:
		test_output.append([0,0,1,0,0,0,0,0,0,0])
	elif test[i][100] == 3:
		test_output.append([0,0,0,1,0,0,0,0,0,0])
	elif test[i][100] == 4:
		test_output.append([0,0,0,0,1,0,0,0,0,0])
	elif test[i][100] == 5:
		test_output.append([0,0,0,0,0,1,0,0,0,0])
	elif test[i][100] == 6:
		test_output.append([0,0,0,0,0,0,1,0,0,0])
	elif test[i][100] == 7:
		test_output.append([0,0,0,0,0,0,0,1,0,0])
	elif test[i][100] == 8:
		test_output.append([0,0,0,0,0,0,0,0,1,0])
	elif test[i][100] == 9:
		test_output.append([0,0,0,0,0,0,0,0,0,1])


x_data = np.array(train_input)
y_data = np.array(train_output)



X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
W1 = tf.Variable(tf.random_uniform([100, 256], -1., 1.))
W2 = tf.Variable(tf.random_uniform([256, 256], -1., 1.))
W3 = tf.Variable(tf.random_uniform([256, 10], -1., 1.))

b1 = tf.Variable(tf.zeros([256]))
b2 = tf.Variable(tf.zeros([256]))
b3 = tf.Variable(tf.zeros([10]))
L1 = tf.add(tf.matmul(X, W1), b1)
L1 = tf.nn.relu(L1)
L2 = tf.add(tf.matmul(L1, W2), b2)
L2 = tf.nn.relu(L2)
model = tf.add(tf.matmul(L2, W3), b3)
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=model))
optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
train_op = optimizer.minimize(cost)

init = tf.global_variables_initializer()
#config = tf.ConfigProto(allow_soft_placement = True, log_device_placement=True)
#sess = tf.Session(config = config)
#sess = tf.Session(config=tf.ConfigProto(log_device_placement=True))
# saver = tf.train.Saver()
# saver.save(sess, 'my_test_model')

sess = tf.Session()
sess.run(init)

# print(x_data.shape)
# print(np.array(test_input).shape)

# for step in range(10):
# 	sess.run(train_op, feed_dict={X: x_data, Y: y_data})
# 	# print(x_data)
# 	if (step + 1) % 10 == 0:
# 		print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))


epoch_size = 10
batch_size = 100

for epoch in range(epoch_size):
	batch_count = int(x_data.shape[0]/batch_size)
	for i in range(batch_count):
		batch_xs, batch_ys = x_data[i*batch_size : i*batch_size+batch_size], y_data[i*batch_size: i*batch_size+batch_size]
		sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys})
		if(i+1) % 10 == 0:
			print("epoch:", epoch, "iteration:", i, sess.run(cost, feed_dict={X:batch_xs, Y: batch_ys}))


#########
# 결과 확인
#########

##### training data
prediction = tf.argmax(model, 1)
target = tf.argmax(Y, 1)
# print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
# print('실제값:', sess.run(target, feed_dict={Y: y_data}))
sess.run(prediction, feed_dict={X: x_data})
sess.run(target, feed_dict={Y: y_data})
# for i in range(0,100):
# 	print(sess.run(prediction[i]))
# 	print(sees.run(target[i]))
# 	print("=================")
is_correct = tf.equal(prediction, target)
# print(sess.run(is_correct[0]))
# print("is_correct")
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('예측값:', sess.run(prediction, feed_dict={X: x_data[:30]}))
print('실제값:', sess.run(target, feed_dict={Y: y_data[:30]}))
print('trainin_set 정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))

##### test data
x_data_test = test_input
y_data_test = test_output
# print('예측값:', sess.run(prediction, feed_dict={X: x_data_test[:30]}))
# print('실제값:', sess.run(target, feed_dict={Y: y_data_test[:30]}))
sess.run(prediction, feed_dict={X: x_data_test})
sess.run(target, feed_dict={Y: y_data_test})
is_correct = tf.equal(prediction, target)
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
print('test_set 정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data_test, Y: y_data_test}))
