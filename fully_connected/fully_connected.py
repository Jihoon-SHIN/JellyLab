# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import os as os
import tool as tool

tf.logging.set_verbosity(tf.logging.INFO)


def weight_init(shape):
    initial = tf.truncated_normal(shape, stddev=0.01)
    return tf.Variable(initial);


def bias_init(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def custom_model_fn(features, labels, mode):
    """Model function for PA1"""

    # Write your custom layer
    # Input Layer
    # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1]) # You also can use 1 x 784 vector

    # Implemented by Jihoon #
    input_layer = tf.reshape(features["x"], [-1, 100])
    # Layer 1
    w_fc1 = weight_init([100, 256])
    b_fc1 = bias_init([256])
    h_fc1 = tf.nn.leaky_relu(tf.matmul(input_layer, w_fc1) + b_fc1)  # Using LeakyReLU

    w_fc2 = weight_init([256, 512])
    b_fc2 = bias_init([512])
    h_fc2 = tf.nn.leaky_relu(tf.matmul(h_fc1, w_fc2) + b_fc2)  # Using LeakyReLU

    w_fc3 = weight_init([512, 512])
    b_fc3 = bias_init([512])
    h_fc3 = tf.nn.leaky_relu(tf.matmul(h_fc2, w_fc3) + b_fc3)  # Using LeakyReLU


    # Output logits Layer , the last layer
    logits = tf.layers.dense(inputs=h_fc3, units=10)

    predictions = {
        # Generate predictions (for PREDICT and EVAL mode)
        "classes": tf.argmax(input=logits, axis=1),
        # Add `softmax_tensor` to the graph. It is used for PREDICT and by the
        # `logging_hook`.
        "probabilities": tf.nn.softmax(logits, name="softmax_tensor")
    }

    # In predictions, return the prediction value, do not modify
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    # Select your loss and optimizer from tensorflow API
    # Calculate Loss (for both TRAIN and EVAL modes)
    loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)  # Refer to tf.losses

    # Configure the Training Op (for TRAIN mode)
    if mode == tf.estimator.ModeKeys.TRAIN:
        optimizer = tf.train.AdamOptimizer(learning_rate=0.001)  # Refer to tf.train, Using AdamOptimizer
        train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

    # Add evaluation metrics (for EVAL mode)
    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels=labels, predictions=predictions["classes"])}
    return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)

def pprint(arr):
    print("type:{}".format(type(arr)))
    print("shape: {}, dimension: {}, dtype:{}".format(arr.shape, arr.ndim, arr.dtype))
    print("Array's Data:\n", arr)

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

if __name__ == '__main__':

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


    word_to_ix, ix_to_word = tool.make_dict_all_cut(contents, 0, 3, jamo_delete=True)
    encoder_size = 100
    decoder_size = 3
    encoderinputs, decoderinputs, targets_, targetweights = \
    tool.make_inputs(contents, title, word_to_ix,
                     encoder_size=encoder_size, decoder_size=decoder_size, shuffle=False)

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
        train_output.append(train[i][100])
    for i in range(0, len(test)):
        test_input.append(test[i][:100])
        test_output.append(test[i][100])
    #
    dataset_train = np.array(train_input).astype(np.int32)
    dataset_eval = np.array(test_input).astype(np.int32)
    test_data = np.array(test_input).astype(np.float32)
    eval_data = np.array(test_input).astype(np.float32)
    train_data = dataset_train.astype(np.float32)

    for i in range(10):
        print(train_data[i])
    train_labels = np.array(train_output).astype(np.int32)
    eval_labels = np.array(test_output).astype(np.int32)

    dataset_train = open(os.path.expanduser('train.npy'))
    dataset_train = np.load(dataset_train)
    dataset_eval = open(os.path.expanduser('valid.npy'))
    dataset_eval = np.load(dataset_eval)
    test_data = open(os.path.expanduser('test.npy'))
    test_data = np.load(test_data)
    train_data = dataset_train[:, :784]


    train_labels = dataset_train[:, 784].astype(np.int32)
    eval_labels = dataset_eval[:, 784].astype(np.int32)
    
    # Save model and checkpoint
    classifier = tf.estimator.Estimator(model_fn=custom_model_fn, model_dir="./model")
    
    # Set up logging for predictions
    tensors_to_log = {"probabilities": "softmax_tensor"}
    logging_hook = tf.train.LoggingTensorHook(tensors=tensors_to_log, every_n_iter=50)
    
    # Train the model. You can train your model with specific batch size and epoches
    train_input = tf.estimator.inputs.numpy_input_fn(x={"x": train_data}, y=train_labels, batch_size=100,
                                                     num_epochs=None, shuffle=True)
    classifier.train(input_fn=train_input, steps=20000, hooks=[logging_hook])
    
    # Eval the model. You can evaluate your trained model with validation data
    eval_input = tf.estimator.inputs.numpy_input_fn(x={"x": eval_data}, y=eval_labels, num_epochs=1, shuffle=False)
    eval_results = classifier.evaluate(input_fn=eval_input)
    
    ## ----------- Do not modify!!! ------------ ##
    # Predict the test dataset
    pred_input = tf.estimator.inputs.numpy_input_fn(x={"x": test_data}, shuffle=False)
    pred_results = classifier.predict(input_fn=pred_input)
    result = np.asarray([x.values()[1] for x in list(pred_results)])
