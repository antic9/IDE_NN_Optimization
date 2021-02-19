# -*- coding: utf-8 -*-
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
from tensorflow.examples.tutorials.mnist import input_data

import IDE_accuracy
import IDE_crossentropy
import Individual
import JADE_accuracy
import JADE_crossentropy
import jDE_entropy
import jDE_accuracy
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import tensorflow as tf

#parameters

#メモ:試行回数を引数から呼び出して、その回数実行するように仕様変更、最後に配列から平均、中央値、標準偏差をだす
GENERATION = 1000
POPULATION = 500
COUNTER = 10
DIMENSION = 7850
batch_size = 200
fitness_value = "accuracy"
# fitness_value = "entropy"
# Method = "IDE" 
# Method = "jDE"
Method = "JADE"

if __name__ == '__main__':
    args = sys.argv

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
    un_update_counter = 0
    
    batch_xs, batch_ys = mnist.train.next_batch(200)
    # de = jDE.adaptive_DE(batch_xs, batch_ys)
    # de = JADE.JADE(batch_xs,batch_ys)
    if(Method == "jDE"):
        if(fitness_value == "accuracy"):
            de = jDE_accuracy.adaptive_DE(batch_xs, batch_ys, POPULATION)
        else:
            de = jDE_entropy.adaptive_DE(batch_xs, batch_ys, POPULATION)
    if(Method == "JADE"):
        if(fitness_value == "accuracy"):
            de = JADE_accuracy.JADE(batch_xs, batch_ys,POPULATION)
        else:
            de = JADE_crossentropy.JADE(batch_xs, batch_ys,POPULATION)
    # print(type(batch_xs))
    elif(Method == "IDE"):
        if(fitness_value == "accuracy"):
            de = IDE_accuracy.IDE_NN(batch_xs, batch_ys,POPULATION)
        else:
            de = IDE_crossentropy.IDE_NN(batch_xs, batch_ys,POPULATION)
    # scipy.stats.zscore(a(行列),axis(正規化行ないたい次元),ddof(自由度))
    # norm_batch_xs = scipy.stats.zscore(batch_xs,None)
    # print(batch_xs[0])
    # print(batch_ys[0])
    # print(mnist.test.labels.shape)
    # print(type(batch_xs))
    # print(batch_xs.shape)

    # print("normalized")
    # print(norm_batch_xs[0])
    indiv = Individual.Indivi(batch_xs, batch_ys)
    # indiv = Individual.Indivi(norm_batch_xs, batch_ys)

    
    # print(batch_xs.shape())
    print(batch_ys)
    #main program
    for i in range(GENERATION):
        euclidlist =[]
        batch_xs, batch_ys = mnist.train.next_batch(200)
        # norm_batch_xs = scipy.stats.zscore(batch_xs,None)
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
        de.Best_plot(Bestindividual,Bestgene,Bestaccuracy,mnist.test.images,mnist.test.labels)
               
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
    name_str = "_p"+str(POPULATION)+"_g"+str(GENERATION)+"_b"+str(batch_size)+"_MAGIC_"+fitness_value
    if(Method=="jDE"):
        # with open('jde_weight'+number+'.csv', 'w+',newline='') as f:
        #     w = csv.writer(f)
        #     w = w.writerows(Bestgene)
        # with open('jde_Accuracy'+number+'.csv', 'w+') as f:
        #     w = csv.writer(f)
        #     w = w.writerow(Bestaccuracy)
        # with open('jde_crossentropy'+number+'.csv', 'w+',newline='') as f:
        #     w = csv.writer(f)
        #     w = w.writerow(Bestindividual)
        # with open('jde_weight'+name_str+'.csv', 'a+',newline='') as f:
        #     w = csv.writer(f,lineterminator = "\n")
        #     w = w.writerows(Bestgene)
        with open('Result_data/jde_Accuracy'+name_str+'.csv', 'a+') as f:
            w = csv.writer(f,lineterminator = "\n")
            w = w.writerow(Bestaccuracy)
        with open('Result_data/jde_crossentropy'+name_str+'.csv', 'a+',newline='') as f:
            w = csv.writer(f,lineterminator = "\n")
            w = w.writerow(Bestindividual)
    elif(Method == "JADE"):
        # with open('jade_weight'+name_str+'.csv', 'a',newline='') as f:
        #     w = csv.writer(f)
        #     w = w.writerows(Bestgene)
        with open('Result_data/jade_Accuracy'+name_str+'.csv', 'a') as f:
            w = csv.writer(f)
            w = w.writerow(Bestaccuracy)
        with open('Result_data/jade_crossentropy'+name_str+'.csv', 'a',newline='') as f:
            w = csv.writer(f)
            w = w.writerow(Bestindividual)
    elif(Method == "IDE"):
        # with open('ide_weight_A'+name_str+'.csv', 'a',newline='') as f:
        #     w = csv.writer(f)
        #     w = w.writerows(Bestgene)
        with open('Result_data/ide_Accuracy_A'+name_str+'.csv', 'a') as f:
            w = csv.writer(f)
            w = w.writerow(Bestaccuracy)
        with open('Result_data/ide_crossentropy_A'+name_str+'.csv', 'a',newline='') as f:
            w = csv.writer(f)
            w = w.writerow(Bestindividual)
    indiv.Finish(Bestgene[i])
    print(Method)
    print(de)
    
        
        
        
        
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

