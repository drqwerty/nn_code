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
x = tf.placeholder("float", [None, 784])  # samples 28x28 pixels
y_ = tf.placeholder("float", [None, 10])  # labels  numbers

# 784 entries, 5 neurons
W1 = tf.Variable(np.float32(np.random.rand(784, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# 5 entries, 10 neurons
W2 = tf.Variable(np.float32(np.random.rand(5, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
# h = tf.matmul(x, W1) + b1  # Try this!
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)  # learning rate: 0.01

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)

print("----------------------")
print("   Start training...  ")
print("----------------------")

batch_size = 20

train_values = []
train_error_values = []

valid_values = []
valid_error_values = []

current_error = 1.
last_error = 0.
epoch = 0

while (abs(current_error - last_error)) > 0.01:
    epoch += 1
    for jj in range(len(train_x) // batch_size):  # Este bucle se ejecuta 5
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # validation
    current_error = sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    valid_error_values.append(current_error)
    if epoch > 1:
        last_error = valid_error_values[-2]
    print("Epoch #:", epoch, "Error: ", current_error)

    current_error2 = sess.run(loss, feed_dict={x: valid_x, y_: valid_y}) / 100
    valid_values.append(current_error2)
    print("Epoch #:", epoch, "Error2: ", current_error2)

    # sess.run(y, feed_dict={x: valid_x})
    # result = sess.run(y, feed_dict={x: valid_x})
    # valid_values.append(result)
    # for b, r in zip(valid_y, result):
    #     print(b, "-->", r)
    # print("----------------------------------------------------------------------------------")

print("------------------")
print("   Start test...  ")
print("------------------")

error = 0.
good = 0.
result = sess.run(y, feed_dict={x: test_x})
for b, r in zip(test_y, result):
    # print(b, "-->", r)
    # print(np.argmax(b), "-->", np.argmax(r))
    if (np.argmax(b) != np.argmax(r)):
        error += 1
    else:
        good += 1
print('Error: %.2f%%' % (error / len(result) * 100))
print('Good: %.2f%%' % (good / len(result) * 100))

plt.title("Ej: titulo")
plt.plot(valid_error_values)
plt.plot(valid_values)
plt.show()