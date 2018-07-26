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


model = seq2seq(multi=multi, hidden_size=hidden_size, num_layers=num_layers,
learning_rate=learning_rate, batch_size=batch_size,
vocab_size=vocab_size, 
encoder_size=encoder_size, 
decoder_size=decoder_size,
forward_only=forward_only)

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


step_time, loss = 0.0, 0.0
current_step = 0
start = 0
end = batch_size
while current_step < 14553:
while current_step<int(len(title)*(1/batch_size)*0.8*epoch):
# while current_step<100:
    # if end > len(title):
    if end > len(title)*0.8:
        start = 0
        end = batch_size
    # Get a batch and make a step
    start_time = time.time()
    # for i in range(start, end):
    # tmp_index_list = []
    encoderinputs_tmp = []
    decoderinputs_tmp = []
    targets__tmp = []
    targetweights_tmp = []
    title_tmp = []
    # print(mix_index[start:end])
    for index in mix_index[start:end]:
        # tmp_index_list.append(index)
        encoderinputs_tmp.append(encoderinputs[index])
        decoderinputs_tmp.append(decoderinputs[index])
        targets__tmp.append(targets_[index])
        targetweights_tmp.append(targetweights[index])
        title_tmp.append(title[index])
    # encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs[start:end],
    #                                                                           decoderinputs[start:end],
    #                                                                           targets_[start:end],
    #                                                                           targetweights[start:end])
    encoder_inputs, decoder_inputs, targets, target_weights = tool.make_batch(encoderinputs_tmp,
                                                                              decoderinputs_tmp,
                                                                              targets__tmp,
                                                                              targetweights_tmp)
    if current_step % steps_per_checkpoint == 0:
        for i in range(decoder_size - 2):
            decoder_inputs[i + 1] = np.array([word_to_ix['<PAD>']] * batch_size)
        output_logits = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, True)
        predict = [np.argmax(logit, axis=1)[0] for logit in output_logits]
        # predict = ' '.join(ix_to_word[ix][0] for ix in predict)
        predict_v = ix_to_word[predict[0]]
        real = [word[0] for word in targets]
        # real = ' '.join(ix_to_word[ix][0] for ix in real)
        real = ix_to_word[real[0]]
        print('\n----\n step : %s \n time : %s \n LOSS : %s \n 예측 : %s \n 손질한 정답 : %s \n 정답 : %s \n----' %
              (current_step, step_time, loss, predict_v, real, title_tmp[0]))
        if current_step is not 0:
            correct_count = 0
            for i in range(0,len(np.argmax(output_logits[0], axis=1))):
                print(ix_to_word[np.argmax(output_logits[0], axis=1)[i]] + ' ' + title_tmp[i])
                if title_tmp[i] == "직장생활":
                    if ix_to_word[np.argmax(output_logits[0], axis=1)[i]] == "직장생":
                        correct_count += 1    
                else:
                    if ix_to_word[np.argmax(output_logits[0], axis=1)[i]] == title_tmp[i]:
                        correct_count += 1
            print("정확도 : %s, 일치한 개수 : %s" %(correct_count/batch_size, correct_count))
        loss, step_time = 0.0, 0.0
    step_loss = model.step(sess, encoder_inputs, decoder_inputs, targets, target_weights, False)
    step_time += time.time() - start_time / steps_per_checkpoint
    loss += np.mean(step_loss) / steps_per_checkpoint
    current_step += 1
    start += batch_size
    end += batch_size
model.save(sess, checkpoint_prefix, model.global_step.eval(sess))
