from sklearn.model_selection import train_test_split
from sklearn import datasets,svm,metrics
from sklearn.metrics import accuracy_score
import pandas as pd
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf
import numpy as np
import csv
import random


x = tf.placeholder(tf.float32, [None, 41])
W = tf.Variable(tf.zeros([41, 2]))
b = tf.Variable(tf.zeros([2]))
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])

cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

init = tf.global_variables_initializer()

sess = tf.Session()
sess.run(init)
accuracy_ = []
cross_entropy_ = []
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


df = pd.read_csv("biodeg_1.csv",sep=";",encoding="utf-8")
x = df.drop(columns = "experimentalclass")
temp = df["experimentalclass"]
temp = temp.replace("RB",1)
y = temp.replace("NRB",0)
# df["expirementalclass"] = pd.Categorical(df["experimentalclass"])
# df["experimentalclass"] = df["experimentalclass"].astype("category")
# df["expirementalclass"] = df.experimentalclass.cat.codes
# print(df.head())
# print(df["expirementalclass"])
# print(x)
# print(y)
dataset = tf.data.Dataset.from_tensor_slices((x.values, y.values))
train_dataset = dataset.shuffle(len(df)).batch(1)
for i in range(1000):
    batch_xs, batch_ys = train_dataset.train.next_batch(100)
    sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})
print(sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels}))
# sess.run(init)
# print(sess.run(dataset))


