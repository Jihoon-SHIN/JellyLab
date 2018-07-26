# -*- coding: utf-8 -*-
import os
import time
import datetime
from tensorflow import flags
import tensorflow as tf
import numpy as np
from tensorflow.contrib import layers


class TextCNN(object):
    # A CNN for text classification
    # Uses an embedding layer, followed by a convolutional, max-pooling and softmax layer.
    #
    # <Parameters>
    # 	-sequence_length : the length of our sentences
    # 	-num_classes : Number of classes in the output layer. (e.g two case -> positive and negative)
    # 	-vocab_size : the size of our vocabularay
    # 	-embedding_size : 각 단어에 해당되는 embedded vector의 차원
    # 	-filter_sizes : convolutional filter들의 사이즈 (= 각 filter가 몇 개의 단어를 볼 것 인가?)
    # 	-num_filters : 각 filter size 별 filter 수
    # 	-l2_reg_lambda : 각 weights, biases에 대한 l2 regularization 정도

    def __init__(self, sequence_length, num_classes, vocab_size, embedding_size,
                 filter_sizes, num_filters, l2_reg_lambda=0.0):
        # Placeholders for input, output and dropout
        self.input_x = tf.placeholder(tf.int32, [None, sequence_length], name="input_x")
        self.input_y = tf.placeholder(tf.float32, [None, num_classes], name="input_y")
        self.dropout_keep_prob = tf.placeholder(tf.float32, name="dropout_keep_prob")

        # Keeping track of 12 regularization loss (optional)
        l2_loss = tf.constant(0.0)

        # Embedding layer

        #
        # <Variable>
        # 	- W : 각 단어ㅢ 임베디드 벡터의 성분을 랜덤하게 할당

        # The first layer we define is the embedding layer, which maps vocabulary word indices into low-dimensional vector representations.
        # with tf.device('/gpu:0'), tf.name_scope("embedding")
        with tf.device('/cpu:0'), tf.name_scope("embedding"):
            W = tf.Variable(
                tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0), name="W")
            # embedding_lookup 은 sentence 길이만큼 잘라주는 역할을 함
            self.embedded_chars = tf.nn.embedding_lookup(W, self.input_x)
            self.embedded_chars_expanded = tf.expand_dims(self.embedded_chars, -1)

        # Create a convolution + maxpool layer for each filter size
        pooled_outputs = []
        for i, filter_size in enumerate(filter_sizes):
            with tf.name_scope("conv-maxpool-%s" % filter_size):
                # Convolution Layer
                filter_shape = [filter_size, embedding_size, 1, num_filters]
                W = tf.Variable(tf.truncated_normal(filter_shape, stddev=0.1), name="W")
                b = tf.Variable(tf.constant(0.1, shape=[num_filters]), name="b")
                conv = tf.nn.conv2d(
                    self.embedded_chars_expanded,
                    W,
                    strides=[1, 1, 1, 1],
                    padding="VALID",
                    name="conv"
                )
                # Apply nonlinearity
                h = tf.nn.relu(tf.nn.bias_add(conv, b), name="relu")
				# Maxpooling over the ouputs
                pooled = tf.nn.max_pool(
                    h,
                    ksize=[1, sequence_length-filter_size+1, 1, 1],
                    strides=[1,1,1,1],
                    padding='VALID',
                    name="pool"
                )
                pooled_outputs.append(pooled)
        # Combine all the pooled features
        num_filters_total = num_filters * len(filter_sizes)
        self.h_pool = tf.concat(pooled_outputs,3)
        self.h_pool_flat = tf.reshape(self.h_pool, [-1, num_filters_total])

        # Add dropout
        with tf.name_scope("dropout"):
            self.h_drop = tf.nn.dropout(self.h_pool_flat, self.dropout_keep_prob)

        # Final (unnormalized) scores and predictions
        with tf.name_scope("output"):
            W = tf.get_variable(
                "W",
                shape=[num_filters_total, num_classes],
                initializer=tf.contrib.layers.xavier_initializer())
            b = tf.Variable(tf.constant(0.1, shape=[num_classes]), name="b")
            l2_loss += tf.nn.l2_loss(W)
            l2_loss += tf.nn.l2_loss(b)
            self.scores = tf.nn.xw_plus_b(self.h_drop, W, b, name="scores")
            self.predictions = tf.argmax(self.scores, 1, name="predictions")
        # Calculate Mean cross-entropy lostt
        with tf.name_scope("loss"):
            losses = tf.nn.softmax_cross_entropy_with_logits(logits=self.scores, labels=self.input_y)
            self.loss = tf.reduce_mean(losses) + l2_reg_lambda + l2_loss
        # Accuracy
        with tf.name_scope("accuracy"):
            correct_predictions = tf.equal(self.predictions, tf.argmax(self.input_y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_predictions, "float"), name="accuracy")