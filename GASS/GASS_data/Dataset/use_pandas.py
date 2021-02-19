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


x = tf.placeholder(tf.float32, [None, 128])
W = tf.Variable(tf.zeros([128, 6]))
b = tf.Variable(tf.zeros([6]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 6])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
accuracy_ = []
index = 0
batch_size = 100
cross_entropy_ = []
GASS_all_list = []
GASS_input_all = []
GASS_output_all = []
GASS_input = []
GASS_output = []
GASS_input_test = []
GASS_output_test = []
input_batch = np.zeros((batch_size,128))
output_batch = np.zeros((batch_size,6))

correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

dat = "batch1.dat"
df = pd.read_table(dat,sep="\s+",index_col=0,header=None)
for c in df.columns.values:
    df[c] = df[c].apply(lambda x: float(str(x).split(':')[1]))
dfs = df
for i in range(2,4):
    print(i)
    dat2 = "batch"+str(i)+".dat"
    df2 = pd.read_table(dat2,sep="\s+",index_col=0,header=None)
    for c in df2.columns.values:
        df2[c] = df2[c].apply(lambda x: float(str(x).split(':')[1]))    
    dfs = dfs.append(df2)
dat2 = "batch4.dat"
dftest = pd.read_table(dat2,sep="\s+",index_col=0,header=None)
for c in df2.columns.values:
    dftest[c] = dftest[c].apply(lambda x: float(str(x).split(':')[1]))    
# dftest = dftest.append(dftest)
for i in range(5,11):
    print(i)
    dat2 = "batch"+str(i)+".dat"
    df2 = pd.read_table(dat2,sep="\s+",index_col=0,header=None)
    for c in df2.columns.values:
        df2[c] = df2[c].apply(lambda x: float(str(x).split(':')[1]))    
    dfs = dfs.append(df2)
GASS_output_all = dfs.index.values
GASS_input_all = dfs.values
GASS_input_test = dftest.values
GASS_output_test = dftest.index.values
for count in range(len(GASS_input_all)):
    temp = []
    temp.append(GASS_input_all[count])
    temp.append(GASS_output_all[count])
    GASS_all_list.append(temp)
GASS_all = np.array(GASS_all_list)
np.random.shuffle(GASS_all)

for t3 in range(batch_size):
    if(index == len(GASS_input_all)):
        index = 0
    input_batch[t3] = GASS_all[index][0]
    answer = GASS_all[index][1]
    output_batch[t3][answer-1] = 1
    index = index + 1
print(len(GASS_all))
for i in range(1000):
    batch_xs = input_batch
    batch_ys = output_batch
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
    for t3 in range(batch_size):
        if(index == len(GASS_input_all)):
            index = 0
        input_batch[t3] = GASS_all[index][0]
        answer = GASS_all[index][1]
        output_batch[t3][answer-1] = 1
        index = index + 1
testbatch_size = len(GASS_input_test)
testbatch_input = np.zeros((testbatch_size,128))
testbatch_output = np.zeros((testbatch_size,6))
index = 0
# print(GASS_input_test)
print(GASS_input_test)
print(len(GASS_input_test))

print(index)

for t3 in range(len(GASS_output_test)):  
    testbatch_input[t3] = GASS_input_test[t3]
    answer = GASS_output_test[t3]
    testbatch_output[t3][answer-1] = 1
print(testbatch_output[160])
print(sess.run(accuracy, feed_dict={x: testbatch_input, y_: testbatch_output}))


