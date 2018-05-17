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
    o_h = np.zeros((len(x), n))  # raw, column
    o_h[np.arange(len(x)), x] = 1  # put 1 in x column, i think
    return o_h


data = np.genfromtxt('iris.data', delimiter=",")  # iris.data file loading
np.random.shuffle(data)  # we shuffle the data
# x_data = data[:, 0:4].astype('f4')  # the samples are the four first rows of data
# y_data = one_hot(data[:, 4].astype(int), 3)  # the labels are in the last row. Then we encode them in one hot code
#
# print "\nSome samples..."
# for i in range(20):
#     print x_data[i], " -> ", y_data[i]

data70 = int(len(data) * 0.7)
data85 = int(len(data) * 0.85)

x_data_training = data[:data70, 0:4].astype('f4')
y_data_training = one_hot(data[:data70, 4].astype(int), 3)

x_data_validation = data[data70:data85, 0:4].astype('f4')
y_data_validation = one_hot(data[data70:data85, 4].astype(int), 3)

x_data_test = data[data85:, 0:4].astype('f4')
y_data_test = one_hot(data[data85:, 4].astype(int), 3)

x = tf.placeholder("float", [None, 4])  # samples
y_ = tf.placeholder("float", [None, 3])  # labels

# 4 entries, 5 neurons
W1 = tf.Variable(np.float32(np.random.rand(4, 5)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(5)) * 0.1)

# 5 entries, 3 neurons
W2 = tf.Variable(np.float32(np.random.rand(5, 3)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(3)) * 0.1)

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

for epoch in range(100):
    for jj in range(len(x_data_training) // batch_size):  # Este bucle se ejecuta 5
        batch_xs = x_data_training[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = y_data_training[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    # validation
    print("Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys}))

    sess.run(y, feed_dict={x: x_data_validation})
    result = sess.run(y, feed_dict={x: x_data_validation})
    for b, r in zip(y_data_validation, result):
        print(b, "-->", r)
    print("----------------------------------------------------------------------------------")

print("------------------")
print("   Start test...  ")
print("------------------")

error = 0.
result = sess.run(y, feed_dict={x: x_data_test})
for b, r in zip(y_data_test, result):
    print(b, "-->", r)
    if (np.argmax(b) != np.argmax(r)):
        error += 1
print('Error: %.2f%%' % (error / len(result)))
