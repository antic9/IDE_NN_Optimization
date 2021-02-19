# coding:UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import numpy as np
from decimal import *
import math
# from tensorflow.examples.tutorials.mnist import input_data
# mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf


#parameters
DIMENSION = 84

#Minist
MIN_VALUE  = -2 #-2
MAX_VALUE  = 2 #2

MID_VALUE = MAX_VALUE - (( MAX_VALUE - MIN_VALUE) /2)



x = tf.placeholder(tf.float32, [None, 41])
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))

W = tf.placeholder(tf.float32, [41,2])
b = tf.placeholder(tf.float32, [2])
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 2])
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
        
sess = tf.Session()
init = tf.global_variables_initializer()
sess.run(init)
        
#with tf.Session() as sess:
#    init= tf.global_variables_initializer()
#    sess.run(init)
    
        
class Indivi:

    def __init__(self, batch_xs, batch_ys, gene = None, max_value = MAX_VALUE, min_value = MIN_VALUE):
    
        
        if gene is None:
            self.gene = np.random.uniform(-0.2,0.2,(1,DIMENSION))
            #self.gene = np.random.uniform(min_value,max_value,(1,DIMENSION))
            #self.gene = np.full((1,DIMENSION), 0.5)
            #self.gene[0,7840:7850]=0.5
            
        else:
            self.gene = gene
        self.fitness_entropy = self.Evaluate(batch_xs, batch_ys)
        self.fitness_accuracy = self.Evaluate2(batch_xs, batch_ys)
        self.F  = 1.0
        self.CR = 0.5
        

    def Eval_Accuracy(self,batch_xs, batch_ys):
        temp = np.array((self.gene[0,0:82]).reshape(41,2))
        temp2 = np.array(self.gene[0,82:84])
        
        out =sess.run(accuracy, feed_dict={x: main.QSAR_input_test, y_: main.QSAR_output_test, W: temp, b: temp2 })
        
        return(out)

    def Evaluate(self,batch_xs, batch_ys):
        
        temp = np.array((self.gene[0,0:82]).reshape(41,2))
        temp2 = np.array(self.gene[0,82:84])
        # print(batch_xs)
        # print(batch_ys)
        out = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, W: temp, b: temp2 })
        
        return float(out)
        
        
    def Evaluate2(self,batch_xs, batch_ys):
    
        temp = np.array((self.gene[0,0:82]).reshape(41,2))
        temp2 = np.array(self.gene[0,82:84])

        out2 =sess.run(accuracy, feed_dict={x:batch_xs, y_: batch_ys, W: temp, b: temp2 })
         

        return float(out2)


    def Finish(self, best_gene,test_input,test_output):
        print(best_gene)
        temp = np.array((best_gene[0,0:82]).reshape(41,2))
        temp2 = np.array(best_gene[0,82:84])
        # print("--------------------")
        # print(test_input)
        # print(test_output)
        # print(temp.shape)
        # print(temp2.shape)
        out = sess.run(cross_entropy, feed_dict={x: test_input, y_: test_output, W: temp, b: temp2 })
        out2 =sess.run(accuracy, feed_dict={x: test_input, y_: test_output, W: temp, b: temp2 })
        print(out)
        print(out2)
        
        sess.close()
