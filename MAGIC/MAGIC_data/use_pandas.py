from sklearn.model_selection import train_test_split
from sklearn import datasets,svm,metrics
from sklearn.metrics import accuracy_score
import pandas as pd
# from pandas.compat import StringIO
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import csv
import random


x = tf.placeholder(tf.float32, [None, 10])
W = tf.Variable(tf.zeros([10, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
accuracy_ = []
index = 0
batch_size = 100
cross_entropy_ = []
MAGIC_all_list = []
MAGIC_input_all = []
MAGIC_output_all = []
MAGIC_input = []
MAGIC_output = []
MAGIC_input_test = []
MAGIC_output_test = []
input_batch = np.zeros((batch_size,10))
output_batch = np.zeros((batch_size,2))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

dat = "magic04.data"
df = pd.read_table(dat,sep="\s+",index_col=0,header=None)
for c in df.columns.values:
    df[c] = df[c].apply(lambda x: float(str(x).split(',')[1])) 
MAGIC_output_all = df.index.values
MAGIC_input_all = df.values
# print(MAGIC_output_all)
for count in range(len(MAGIC_input_all)):
    temp = []
    temp2 = np.array(MAGIC_output_all[count].split(","))
    temp.append(temp2)
    MAGIC_all_list.append(temp2)
MAGIC_all = np.array(MAGIC_all_list)
# np.random.shuffle(MAGIC_all)
print(MAGIC_all[19019][0])
exit(0)
for t3 in range(batch_size):
    if(index == len(MAGIC_input_all)):
        index = 0
    input_batch[t3] = MAGIC_all[index][0]
    answer = MAGIC_all[index][1]
    output_batch[t3][answer-1] = 1
    index = index + 1
print(len(MAGIC_all))
for i in range(1000):
    batch_xs = input_batch
    batch_ys = output_batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    for t3 in range(batch_size):
        if(index == len(MAGIC_input_all)):
            index = 0
        input_batch[t3] = MAGIC_all[index][0]
        answer = MAGIC_all[index][1]
        output_batch[t3][answer-1] = 1
        index = index + 1
testbatch_size = len(MAGIC_input_test)
testbatch_input = np.zeros((testbatch_size,128))
testbatch_output = np.zeros((testbatch_size,6))
index = 0
# print(MAGIC_input_test)
print(MAGIC_input_test)
print(len(MAGIC_input_test))

print(index)

for t3 in range(len(MAGIC_output_test)):  
    testbatch_input[t3] = MAGIC_input_test[t3]
    answer = MAGIC_output_test[t3]
    testbatch_output[t3][answer-1] = 1
print(testbatch_output[160])
print(sess.run(accuracy, feed_dict={x: testbatch_input, y_: testbatch_output}))