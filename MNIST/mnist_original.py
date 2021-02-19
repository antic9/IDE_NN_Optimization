from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
import tensorflow as tf
import numpy as np
import csv

np.set_printoptions(threshold = np.inf)
x = tf.placeholder(tf.float32, [None, 784])
W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.9).minimize(cross_entropy)
 
init = tf.global_variables_initializer()
 
sess = tf.Session()
sess.run(init)
accuracy_ = []
cross_entropy_ = []
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

for i in range(1000):
    batch_xs, batch_ys = mnist.train.next_batch(200)
    # print(batch_ys)
    # print(mnist.test.labels)
    # exit(0)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    accuracy_.append(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
    cross_entropy_.append(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
    
 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
with open('gdo_crossentropy__learningrate0o9.csv', 'a',newline='') as f:
    w = csv.writer(f)
    w = w.writerow(cross_entropy_)
with open('gdo_accuracy__learningrate0o9.csv', 'a',newline='') as f:
    w = csv.writer(f)
    w = w.writerow(accuracy_)

print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
print(sess.run(cross_entropy, feed_dict = {x: mnist.test.images, y_: mnist.test.labels}))
# print(sess.run(W))

 
 
 
 
print(sess.run(b))

arr = np.array([1,1,1,1,1,1,1,1,1,1])

assign_op = b.assign(arr)
print(sess.run(assign_op))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))