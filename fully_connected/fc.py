# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os as os
import tool as tool

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

def weight_init(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial)


def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

agony = ["외모", "가족", "학업", "취업", "직장생활", "진로", "친구", "이성", "이웃", "성격"]
contents = []
title = []
count = 0
result = []
for i in range(len(agony)):
    f = open(agony[i]+"_v7.txt", 'r')
    lines = f.readlines()
    for line in lines:
        contents.append(line.split("\n")[0])
    for line in lines:
        title.append(agony[i])
        result.append([i])

word_to_ix, ix_to_word = tool.make_dict_all_cut(contents, 0, 4, jamo_delete=True)
encoder_size = 100
decoder_size = 3
encoderinputs, decoderinputs, targets_, targetweights = \
tool.make_inputs(contents, title, word_to_ix, encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)

mix = []
for i in range(len(encoderinputs)):
	mix.append(encoderinputs[i] + result[i])
        
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

# X = tf.placeholder(tf.float32)
# Y = tf.placeholder(tf.float32)
# W1 = weight_init([100, 256])
# b1 = bias_init([256])
# h_fc1 = tf.nn.leaky_relu(tf.matmul(X, W1) + b1)  # Using LeakyReLU
#
# W2 = weight_init([256, 10])
# b2 = bias_init([10])
# h_fc2 = tf.nn.leaky_relu(tf.matmul(h_fc1, W2) + b2)  # Using LeakyReLU
#
# cost = tf.reduce_mean(
#     tf.nn.softmax_cross_entropy_with_logits(labels=Y, logits=h_fc2))
#
# optimizer = tf.train.AdamOptimizer(learning_rate=0.01)
# train_op = optimizer.minimize(cost)

# saver = tf.train.Saver()
# sess = tf.Session()
# init = tf.global_variables_initializer()
# sess.run(init)
#
# train_epochs = 5
# batch_size = 100
#
# print(x_data.shape)
# for step in range(train_epochs):
#     batch_count = int(x_data.shape[0]/100)
#     for i in range(batch_count):
#         batch_xs, batch_ys = x_data[i*batch_size:i*batch_size+batch_size], y_data[i*batch_size:i*batch_size+batch_size]
#         sess.run(train_op, feed_dict={X: batch_xs, Y: batch_ys})
#         if (step + 1) % 10 == 0:
#             print(step + 1, sess.run(cost, feed_dict={X: x_data, Y: y_data}))
#
# # saver.save(sess, 'my_test_model')
#
#
# ########
# # 결과 확인 #
# ########
# prediction = tf.argmax(h_fc2, 1)
# target = tf.argmax(Y, 1)
# # print('예측값:', sess.run(prediction, feed_dict={X: x_data}))
# # print('실제값:', sess.run(target, feed_dict={Y: y_data}))
# sess.run(prediction, feed_dict={X: x_data})
# sess.run(target, feed_dict={Y: y_data})
# is_correct = tf.equal(prediction, target)
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print('trainin_set 정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data, Y: y_data}))
#
# ##### test data
# x_data_test = test_input
# y_data_test = test_output
# print('예측값:', sess.run(prediction, feed_dict={X: x_data_test}))
# print('실제값:', sess.run(target, feed_dict={Y: y_data_test}))
# sess.run(prediction, feed_dict={X: x_data_test})
# sess.run(target, feed_dict={Y: y_data_test})
# is_correct = tf.equal(prediction, target)
# accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))
# print('test_set 정확도: %.2f' % sess.run(accuracy * 100, feed_dict={X: x_data_test, Y: y_data_test}))

