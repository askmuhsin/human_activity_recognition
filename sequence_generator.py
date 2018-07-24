from tensorflow.contrib.layers import flatten
import matplotlib.pyplot as plt
import tensorflow as tf
from get_data import load_data
import numpy as np
import random
import cv2
import sys
import os

loc_select = 2

if loc_select == 1:
    loc_src = "../data/img_preprocessed_set_1/"
    dir_to_save = './data/sequence_gen/rnn_sequence_fc2_1.npy'

if loc_select == 2:
    loc_src = "../data/img_preprocessed_set_2/"
    dir_to_save = './data/sequence_gen/rnn_sequence_fc2_2.npy'

dict_seq = {}
class_nums = 6
class_bins = {0: 'shake_hands',
             1: 'hug',
             2: 'kick',
             3: 'point',
             4: 'punch',
             5: 'push'}

def readImg(file_name):
    img = cv2.imread(file_name)
    img = cv2.resize(img, (32, 32))
    return img

def LeNetSequenceGen(x):
    class_nums = 6
    keep_prob = 1
    # Hyperparameters
    mu = 0
    sigma = 0.1
    layer_depth = {
        'layer_1' : 6,
        'layer_2' : 16,
        'layer_3' : 120,
        'layer_f1' : 84
    }

    ## Layer 1: Convolutional. Input = 32x32x1. Output = 28x28x6.
    conv1_w = tf.Variable(tf.truncated_normal(shape = [5,5,1,6],mean = mu, stddev = sigma))
    conv1_b = tf.Variable(tf.zeros(6))
    conv1 = tf.nn.conv2d(x,conv1_w, strides = [1,1,1,1], padding = 'VALID') + conv1_b
    ## Activation.
    conv1 = tf.nn.relu(conv1)

    conv1 = tf.nn.dropout(conv1, keep_prob)

    ## Pooling. Input = 28x28x6. Output = 14x14x6.
    pool_1 = tf.nn.max_pool(conv1,ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    ## Layer 2: Convolutional. Output = 10x10x16.
    conv2_w = tf.Variable(tf.truncated_normal(shape = [5,5,6,16], mean = mu, stddev = sigma))
    conv2_b = tf.Variable(tf.zeros(16))
    conv2 = tf.nn.conv2d(pool_1, conv2_w, strides = [1,1,1,1], padding = 'VALID') + conv2_b
    ## Activation.
    conv2 = tf.nn.relu(conv2)

    conv2 = tf.nn.dropout(conv2, keep_prob)

    ## Pooling. Input = 10x10x16. Output = 5x5x16.
    pool_2 = tf.nn.max_pool(conv2, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'VALID')

    ## Flatten. Input = 5x5x16. Output = 400.
    fc1 = flatten(pool_2)

    ## Layer 3: Fully Connected. Input = 400. Output = 120.
    fc1_w = tf.Variable(tf.truncated_normal(shape = (400,120), mean = mu, stddev = sigma))
    fc1_b = tf.Variable(tf.zeros(120))
    fc1 = tf.matmul(fc1,fc1_w) + fc1_b

    ## Activation.
    fc1 = tf.nn.relu(fc1)

    fc1 = tf.nn.dropout(fc1, keep_prob)

    ## Layer 4: Fully Connected. Input = 120. Output = 84.
    fc2_w = tf.Variable(tf.truncated_normal(shape = (120,84), mean = mu, stddev = sigma))
    fc2_b = tf.Variable(tf.zeros(84))
    fc2 = tf.matmul(fc1,fc2_w) + fc2_b
    ## Activation.
    fc2 = tf.nn.relu(fc2)

    ## cahce vals
    seq_fc2 = fc2

    fc2 = tf.nn.dropout(fc2, keep_prob)

    ## Layer 5: Fully Connected. Input = 84. Output = 6.
    fc3_w = tf.Variable(tf.truncated_normal(shape = (84, class_nums), mean = mu , stddev = sigma))
    fc3_b = tf.Variable(tf.zeros(class_nums))
    logits = tf.matmul(fc2, fc3_w) + fc3_b
    return logits, seq_fc2

def get_x():
    c_n , s_n , f_n = 6, 10, 1000
    for class_num in range(0, c_n):
        for seq_num in range(0, s_n):
            for frame_num in range(0, f_n):
                file_name = loc_src + f"{class_num}_{seq_num}_{frame_num}.png"
                if os.path.isfile(file_name):
                    img = readImg(file_name)
                    img = img[:,:,0]
                    img = ((img-255)/255)
                    test_img = img
                    test_img = test_img.reshape(1, 32, 32, 1)
                    yield test_img, [class_num, seq_num, frame_num]


def writeSequence(temp_seq, temp_seq_meta):
    dict_name = f"class_{temp_seq_meta[0]}"
    nested_dict_name = f"seq_{temp_seq_meta[1]}"
    if dict_name not in dict_seq:
        dict_seq[dict_name] = {}
    temp_seq = np.array(temp_seq)
    dict_seq[dict_name][nested_dict_name] = temp_seq
    print(f"sequence of shape {temp_seq.shape} saved to dict_seq[{dict_name}][{nested_dict_name}]")

def main():
    temp_seq = []
    x = tf.placeholder(tf.float32, (None, 32, 32, 1))
    # keep_prob = tf.placeholder(tf.float32)
    logits = LeNetSequenceGen(x)
    saver = tf.train.Saver()

    with tf.Session() as sess:
        old_seq = 0
        temp_seq_meta = None
        for item in get_x():
            saver.restore(sess, tf.train.latest_checkpoint('./model/'))
            output, seq = sess.run(logits, feed_dict={x: item[0]})
            if not item[1][1]==old_seq:
                old_seq = item[1][1]
                writeSequence(temp_seq, temp_seq_meta)
                temp_seq = []
                temp_seq_meta = None
            temp_seq.append(seq[0])       ## write from fc1 of nn
            # temp_seq.append(output[0])      ## write logits of nn
            temp_seq_meta = item[1]

    writeSequence(temp_seq, temp_seq_meta)
    np.save(dir_to_save, dict_seq)
    print(f"Generated sequence saved at {dir_to_save}")

if __name__ == '__main__':
    main()
