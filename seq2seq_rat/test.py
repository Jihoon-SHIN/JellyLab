# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import tool as tool
import time
import os
from tensorflow.python.client import device_lib
import random


# data loading
agony = ["외모","가족","학업","취업","직장생활","진로","친구","이성","이웃","성격"]
contents = []
title = []
count = 0
for a in agony:
    f = open("/data/"+a+"_v7.txt",'r')
    content = f.readlines()
    count += len(content)
    for i in range(0, len(content)):
        content[i] = content[i].split("\n")[0]
        contents.append(content[i])
    for j in range(0,len(content)):
        title.append(a)
word_to_ix, ix_to_word = tool.make_dict_all_cut(contents, minlength=0, maxlength=3, jamo_delete=True)
# parameters
multi = True
forward_only = False
hidden_size = 256
vocab_size = len(ix_to_word)
num_layers = 3
learning_rate = 0.001
batch_size = 16
encoder_size = 100
decoder_size = 4
# decoder_size = tool.check_doclength(title,sep=True) # (Maximum) number of time steps in this batch
steps_per_checkpoint = 10
epoch = 3

# transform data
encoderinputs, decoderinputs, targets_, targetweights = \
    tool.make_inputs(contents, title, word_to_ix,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)
mix_index = []
for i in range(0, len(contents)):
    mix_index.append(i)
random.shuffle(mix_index)
print(len(mix_index))
# print(mix_index[101])

#To set the directory of model
checkpoint_dir = "./save/checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "rnn")

#To allow to grow gpu memory
sess_config = tf.ConfigProto()
sess_config.gpu_options.allow_growth = True
sess_config.gpu_options.per_process_gpu_memory_fraction = 0.9
sess = tf.Session(config=sess_config)

sess.run(tf.global_variables_initializer())

#To store and reload of model from checkpoint_dir
ckpt = tf.train.get_checkpoint_state(checkpoint_dir)
if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
    print("reload model")
    model.restore(sess, ckpt.model_checkpoint_path)
else:
    print("start from scratch")
    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)




train_size = int(len(title)*0.8) 
test_size = len(title) - train_size
mix_index[:train_size]
encoderinputs_test = []
decoderinputs_test = []
targets__test = []
targetweights_test = []
title_test = []
for index in mix_index[train_size:]:
    # tmp_index_list.append(index)
    encoderinputs_test.append(encoderinputs[index])
    decoderinputs_test.append(decoderinputs[index])
    targets__test.append(targets_[index])
    targetweights_test.append(targetweights[index])
    title_test.append(title[index])
encoderinputs_test_tmp = []
decoderinputs_test_tmp = []
targets__test_tmp = []
targetweights_test_tmp = []
test_all = 0
test_correct = 0
for k in range(0, int(test_size*(1/batch_size))):
    test_all += batch_size
    test_start = k*batch_size
    test_end = k*batch_size + batch_size
    encoderinputs_test_tmp = encoderinputs_test[test_start:test_end]
    decoderinputs_test_tmp = decoderinputs_test[test_start:test_end]
    targets__test_tmp = targets__test[test_start:test_end]
    targetweights_test_tmp = targetweights_test[test_start:test_end]
    title_test_tmp = title_test[test_start:test_end]
    encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs_test_tmp,
                                                                              decoderinputs_test_tmp,
                                                                              targets__test_tmp,
                                                                              targetweights_test_tmp)
    for i in range(decoder_size - 2):
        decoder_inputs[i + 1] = np.array([word_to_ix['<PAD>']] * batch_size)
    output_logits = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, True)

    for i in range(0,len(np.argmax(output_logits[0], axis=1))):
        # print(ix_to_word[np.argmax(output_logits[0], axis=1)[i]] + ' ' + title_tmp[i])
        if title_test_tmp[i] == "직장생활":
            if ix_to_word[np.argmax(output_logits[0], axis=1)[i]] == "직장생":
                test_correct += 1    
        else:
            if ix_to_word[np.argmax(output_logits[0], axis=1)[i]] == title_test_tmp[i]:
                test_correct += 1
        # print("정확도 : %s, 일치한 개수 : %s" %(correct_count/batch_size, correct_count))
print("전체개수 : %s" %test_all)
print("맞춘개수 : %s" %test_correct)
print("정확도 : %s" %(test_correct/test_all))

