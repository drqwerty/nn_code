import gzip
import pickle as cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h


f = gzip.open('mnist.pkl.gz', 'rb')
# 50000, 10000, 10000 len
train_set, valid_set, test_set = cPickle.load(f, encoding='iso-8859-1')
f.close()

train_x, train_y = train_set
train_y = one_hot(train_y, 10)
valid_x, valid_y = valid_set
valid_y = one_hot(valid_y, 10)
test_x, test_y = test_set
test_y = one_hot(test_y, 10)

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

# plt.imshow(train_x[56].reshape((28, 28)), cmap=cm.Greys_r)
# plt.show()  # Let's see a sample
# print(train_y[56])

# TODO: the neural net!!
# x = tf.placeholder("float", [None, 784])  # samples 28x28 pixels
# y_ = tf.placeholder("float", [None, 10])  # labels  numbers


import tensorflow as tf

x = tf.placeholder("float", None)
y = x * 2

with tf.Session() as session:
    result = session.run(y, feed_dict={x: [1, 2, 3]})
    result = session.run(y, feed_dict={x: [4, 5, 6]})
    print(result)

print(result)