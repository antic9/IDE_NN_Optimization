# -*- coding: utf-8 -*-
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tensorflow.examples.tutorials.mnist import input_data

import IDE
import Indi
import JADE
import jDE
import IDE_revised

mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#parameters

#メモ:試行回数を引数から呼び出して、その回数実行するように仕様変更、最後に配列から平均、中央値、標準偏差をだす
GENERATION = 1000
POPULATION = 100
COUNTER = 10
DIMENSION = 7850

if __name__ == '__main__':
    args = sys.argv
    np.set_printoptions(threshold=np.inf)
    #define parameters
    count = 0.0
    Avarage_fitness = []
    Bestaccuracy = []
    Bestindividual = []
    Bestgene = []
    UpdatedIndividual = []
    ps2=0.1
    euclidgraph = []
    euclidavelist = []
    r_glist=[]
    pslist = []
    zero_dim = []
    un_update_counter = 0
    batch_xs, batch_ys = mnist.train.next_batch(100)
    for k in range(batch_xs.shape[1]):
        z_count = 0
        for j in range(batch_xs.shape[0]):
            if batch_xs[j,k] == 0:
                z_count += 1
        if z_count == batch_xs.shape[0]:
            zero_dim.append(k)
    print(zero_dim)
    # exit(0)
    # de = jDE.adaptive_DE(batch_xs, batch_ys)
    # de = JADE.JADE(batch_xs,batch_ys)
    de = IDE_revised.IDE_NN(batch_xs, batch_ys)
    # scipy.stats.zscore(a(行列),axis(正規化行ないたい次元),ddof(自由度))
    # norm_batch_xs = scipy.stats.zscore(batch_xs,None)
    # print(batch_xs[0])
    # print(batch_ys[0])
    # print(mnist.test.labels.shape)
    # print(type(batch_xs))
    # print(batch_xs.shape)

    # print("normalized")
    # print(norm_batch_xs[0])
    indiv = Indi.Indivi(batch_xs, batch_ys,zero_dim)
    # indiv = Individual.Indivi(norm_batch_xs, batch_ys)

    
    # print(batch_xs.shape())
    # print(batch_ys)
    #main program
    for i in range(GENERATION):
        euclidlist =[]
        batch_xs, batch_ys = mnist.train.next_batch(200)
        # norm_batch_xs = scipy.stats.zscore(batch_xs,None)
        zero_dim= []
        for k in range(batch_xs.shape[1]):
                z_count = 0
                for j in range(batch_xs.shape[0]):
                    if batch_xs[j,k] == 0:
                        z_count += 1
                if z_count == batch_xs.shape[0]:
                    zero_dim.append(k)
        print("Generation="+str(i))

        count += 1.0
        
        """
        de.euclid(euclidlist,euclidgraph,euclidavelist)
        #de.euclid(euclidlist,euclidgraph)
        euclidlistvar = np.var(euclidlist)
        psdenom = np.log10(euclidlistvar)
        if (i==0): r_1=np.array(euclidavelist[0])
        r_g = np.array(euclidavelist[i])
        r_glist.append(r_g)
        
        if r_1 > r_g : ps2 = 0.1 + 0.9*(1-r_g/r_1)
        else:  ps2=ps2
        print(ps2)
        pslist.append(ps2)
        """

        
        #for j in range(10):
        de.Mutation(count,ps2, UpdatedIndividual, batch_xs, batch_ys)
        # de.Mutation(count,ps2, UpdatedIndividual, norm_batch_xs, batch_ys)
        
        de.Best_Individual()
        Avarage_fitness.append(de.Average_Fitness())
        de.Best_plot(Bestindividual,Bestgene,Bestaccuracy)
               
        if (UpdatedIndividual[i] == 0):
            un_update_counter += 1
        if (un_update_counter >= COUNTER ):
            for p in range(len(jDE.Population)):
                if (p == 0) : gene = Bestgene[i]
                else: gene = np.random.uniform(-0.2,0.2,(1,DIMENSION))
                new_Individual = Individual.Indivi(batch_xs, batch_ys, gene)
                jDE.Population[p] = new_Individual
            print("!---reset---!")
            un_update_counter = 0
    print(type(Bestaccuracy[0]))
    with open('Result_data/weight_revisedIDE_g1000_p50.csv', 'a',newline='') as f:
        w = csv.writer(f)
        w = w.writerows(Bestgene)
    with open('Result_data/Accuracy_revisedIDE_g1000_p50.csv', 'a') as f:
        w = csv.writer(f)
        w = w.writerow(Bestaccuracy)
    with open('Result_data/crossentropy_revisedIDE_g1000_p50.csv', 'a',newline='') as f:
        w = csv.writer(f)
        w = w.writerow(Bestindividual)
    indiv.Finish(Bestgene[i])
        
        
        
        
"""
    #↓output↓
    f = open('Results/%s/Best/bestindividual_%s.csv' % (args[1],args[2]), 'w')
    for n in Bestindividual:
        f.write(str(n) + "\n")
    f.close()

    f = open('Results/%s/Update/Update_times_%s.csv' % (args[1],args[2]), 'w')
    for n in UpdatedIndividual:
        f.write(str(n) + "\n")
    f.close()

    f = open('Results/%s/Ps/Ps_%s.csv' % (args[1],args[2]), 'w')
    for n in pslist:
        f.write(str(n) + "\n")
    f.close()
    
    f = open('Results/%s/Rg/Rg_%s.csv' % (args[1],args[2]), 'w')
    for n in r_glist:
        f.write(str(n) + "\n")
    f.close()
"""

