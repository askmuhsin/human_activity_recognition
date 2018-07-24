from tensorflow.contrib.layers import flatten
import tensorflow as tf


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
    return seq_fc2


def main():
    print("helper function to load CNN !")

if __name__ == '__main__':
    main()
