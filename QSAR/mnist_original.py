from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
import tensorflow as tf
import numpy as np
import csv
import random
import convert_qsar

 
x = tf.placeholder(tf.float32, [None, 41])
W = tf.Variable(tf.zeros([41, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.005).minimize(cross_entropy)
 
init = tf.global_variables_initializer()
 
sess = tf.Session()
sess.run(init)
accuracy_ = []
cross_entropy_ = []

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

index = 0
batch_size = 40
QSAR_data= []
QSAR_input = []
QSAR_output = []
QSAR_output_test = []
QSAR_input_test = []

QSAR_input,QSAR_output,QSAR_input_test,QSAR_output_test = convert_qsar.convert_QSAR.Convert()

for i in range(1000):
    for t3 in range(batch_size):
        if(index == len(QSAR_input)):
            index = 0
        temp3 = np.zeros((batch_size,41))
        temp4 = np.zeros((batch_size,2))
        for t4 in range(len(QSAR_input[0])):
            # print(t4)
            temp3[t3][t4] = QSAR_input[index][t4]
        for t5 in range(len(QSAR_output[0])):
            temp4[t3][t5] = QSAR_output[index][t5]
        index = index + 1
    batch_xs = temp3
    batch_ys = temp4
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    accuracy_.append(sess.run(accuracy, feed_dict={x: QSAR_input_test, y_: QSAR_output_test}))
    cross_entropy_.append(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
    
 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
print(sess.run(accuracy, feed_dict={x: QSAR_input_test, y_: QSAR_output_test}))
# print(sess.run(cross_entropy, feed_dict = {x: QSAR_input_test, y_: QSAR_output_test}))

 
 
 
 
# print(sess.run(b))

# arr = np.array([1,1,1,1,1,1,1,1,1,1])

# assign_op = b.assign(arr)
# print(sess.run(assign_op))

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

with open('Result_data/gdo_crossentropy_b40_lr_0dot005.csv', 'a',newline='') as f:
    w = csv.writer(f)
    w = w.writerow(cross_entropy_)
with open('Result_data/gdo_accuracy_b40_lr_0dot005.csv', 'a',newline='') as f:
    w = csv.writer(f)
    w = w.writerow(accuracy_)

