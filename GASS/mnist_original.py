from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
 
import tensorflow as tf
import numpy as np
import csv
import random
import convert_GASS

 
x = tf.placeholder(tf.float32, [None, 128])
W = tf.Variable(tf.zeros([128, 6]))
b = tf.Variable(tf.zeros([6]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 6])
 
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))


train_step = tf.train.GradientDescentOptimizer(0.0001).minimize(cross_entropy)
 
init = tf.global_variables_initializer()
 
sess = tf.Session()
sess.run(init)
accuracy_ = []
cross_entropy_ = []

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

index = 0
batch_size = 50
GASS_data= []
GASS_input = []
GASS_output = []
GASS_output_test = []
GASS_input_test = []

GASS_input,GASS_output,GASS_input_test,GASS_output_test = convert_GASS.convert_GASS.Convert()

for i in range(2000):
    for t3 in range(batch_size):
        if(index == len(GASS_input)):
            index = 0
        temp3 = np.zeros((batch_size,128))
        temp4 = np.zeros((batch_size,6))
        for t4 in range(len(GASS_input[0])):
            # print(t4)
            temp3[t3][t4] = GASS_input[index][t4]
        for t5 in range(len(GASS_output[0])):
            temp4[t3][t5] = GASS_output[index][t5]
        index = index + 1
    batch_xs = temp3
    batch_ys = temp4
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    accuracy_.append(sess.run(accuracy, feed_dict={x: GASS_input_test, y_: GASS_output_test}))
    cross_entropy_.append(sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys}))
    
 
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
print(sess.run(accuracy, feed_dict={x: GASS_input_test, y_: GASS_output_test}))
# print(sess.run(cross_entropy, feed_dict = {x: GASS_input_test, y_: GASS_output_test}))

 
 
 
 
# print(sess.run(b))

# arr = np.array([1,1,1,1,1,1,1,1,1,1])

# assign_op = b.assign(arr)
# print(sess.run(assign_op))

# correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
 
# accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
 
# print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))

with open('gdo_crossentropy.csv', 'a',newline='') as f:
    w = csv.writer(f)
    w = w.writerow(cross_entropy_)
with open('gdo_accuracy.csv', 'a',newline='') as f:
    w = csv.writer(f)
    w = w.writerow(accuracy_)

