# coding:UTF-8
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


import numpy as np
from decimal import *
import math
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf


#parameters
DIMENSION = 7850

#Minist
MIN_VALUE  = -2 #-2
MAX_VALUE  = 2 #2

MID_VALUE = MAX_VALUE - (( MAX_VALUE - MIN_VALUE) /2)



x = tf.placeholder(tf.float32, [None, 784])
#W = tf.Variable(tf.zeros([784, 10]))
#b = tf.Variable(tf.zeros([10]))

W = tf.placeholder(tf.float32, [784,10])
b = tf.placeholder(tf.float32, [10])
y = tf.nn.softmax(tf.matmul(x, W) + b)
y_ = tf.placeholder(tf.float32, [None, 10])
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

    def __init__(self, batch_xs, batch_ys, zero_dim = [],gene = None, max_value = MAX_VALUE, min_value = MIN_VALUE):
        
        
        if gene is None:
            self.gene = np.random.uniform(-0.2,0.2,(1,DIMENSION))
            #self.gene = np.random.uniform(min_value,max_value,(1,DIMENSION))
            #self.gene = np.full((1,DIMENSION), 0.5)
            #self.gene[0,7840:7850]=0.5
            if len(zero_dim)!= 0:
                count = 0
                for i in range(784):
                    if(len(zero_dim) > count):
                        if(i == zero_dim[count]):
                            for j in range(10):
                                self.gene[0,i+j*784] = 0
                                # print("settozero")
                                count += 1
                # print(self.gene)
            
        else:
            self.gene = gene
            if len(zero_dim)!= 0:
                count = 0
                for i in range(784):
                    if(len(zero_dim) > count):
                        if(i == zero_dim[count]):
                            for j in range(10):
                                self.gene[0,i+j*784] = 0
                                # print("settozero")
                                count += 1
        if len(zero_dim)!= 0:
                count = 0
                for i in range(784):
                    if(len(zero_dim) > count):
                        if(i == zero_dim[count]):
                            for j in range(10):
                                self.gene[0,i+j*784] = 0
                                # print("settozero")
                                count += 1
        self.fitness_entropy = self.Evaluate(batch_xs, batch_ys,zero_dim)
        self.fitness_accuracy = self.Evaluate2(batch_xs, batch_ys)
        self.F  = 1.0
        self.CR = 0.5
        

    def Eval_Accuracy(self,batch_xs, batch_ys):
        temp = np.array((self.gene[0,0:7840]).reshape(784,10))
        temp2 = np.array(self.gene[0,7840:7850])
        
        out =sess.run(accuracy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, W: temp, b: temp2 })
        
        return(out)

    def Evaluate(self,batch_xs, batch_ys,zero_dim):
        if len(zero_dim)!= 0:
                count = 0
                for i in range(784):
                    if(len(zero_dim) > count):
                        if(i == zero_dim[count]):
                            for j in range(10):
                                self.gene[0,i+j*784] = 0
                                # print("settozero")
                                count += 1
        temp = np.array((self.gene[0,0:7840]).reshape(784,10))
        temp2 = np.array(self.gene[0,7840:7850])
        
        out = sess.run(cross_entropy, feed_dict={x: batch_xs, y_: batch_ys, W: temp, b: temp2 })
         
        return float(out)
        
        
    def Evaluate2(self,batch_xs, batch_ys):
    
        temp = np.array((self.gene[0,0:7840]).reshape(784,10))
        temp2 = np.array(self.gene[0,7840:7850])

        out2 =sess.run(accuracy, feed_dict={x:batch_xs, y_: batch_ys, W: temp, b: temp2 })
         

        return float(out2)



    def Finish(self, best_gene):
        print(best_gene)
        temp = np.array((best_gene[0,0:7840]).reshape(784,10))
        temp2 = np.array(best_gene[0,7840:7850])
        out = sess.run(cross_entropy, feed_dict={x: mnist.test.images, y_: mnist.test.labels, W: temp, b: temp2 })
        out2 =sess.run(accuracy, feed_dict={x:mnist.test.images, y_: mnist.test.labels, W: temp, b: temp2 })
        print(out)
        print(out2)
        
        sess.close()
