# coding:UTF-8
import Individual
import numpy as np
import operator
import random

import matplotlib
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#from dpgmm2 import DPGMM
matplotlib.use('Agg')
import matplotlib.pyplot as plt



#parameters
POPULATION =  50



F          =  0.5
CR         =  0.5
u_CR       =  0.5
u_F        =  0.5
p2         =  0.05
c          =  0.1
Population = []
MAX_VALUE = Individual.MAX_VALUE
MIN_VALUE = Individual.MIN_VALUE

# tf.reset_default_graph()



class JADE:
    def __init__(self,batch_xs, batch_ys):
        for p in range(POPULATION):
            Population.append(Individual.Indivi(batch_xs, batch_ys))
        self.u_CR=u_CR
        self.u_F=u_F
        self.p2=p2
        self.c=c
        self.POPULATION=POPULATION
        self.A=[]
        self.S_CR=[]
        self.S_F=[]

    def Mutation(self,count,ps2,UpdatedIndividual,batch_xs, batch_ys):
        #Stack_NewIndividual for Scatter plot
        fitness_entropy = []
        fitness_accuracy = []
        GeneX = []
        GeneY = []
        update = 0

        #Mutant Opertion
        
        target_fitness = np.zeros(POPULATION)
        
        
        for i in range(POPULATION):
            for k in range(POPULATION):
                target_fitness[k] = Population[k].fitness_entropy
            current_best_idx = np.argmin(target_fitness)
            p = np.arange(POPULATION)
            #p = np.delete(p,i)
            np.random.shuffle(p)
            # JADE Algorithm. Change F_i and CR_i
            Population[i].CR = np.random.normal(self.u_CR,0.1)
            Population[i].F = np.random.normal(self.u_F,0.1)

            #Select x^p_best individual
            np.random.shuffle(p[:int(self.p2*self.POPULATION)])
            #p_best = p[0]
            p_best = current_best_idx
            #p = np.delete(p,0)
            np.random.shuffle(p)
            p_r1 = p[0]
            p = np.delete(p,0)
            p = np.hstack((p,np.arange(self.POPULATION+len(self.A))))
            np.random.shuffle(p)
            p_r2 = p[0]
            
            x2 = Population[p_r2] if p_r2 < self.POPULATION else self.A[self.POPULATION-p_r2]

            
            
            vector_xx = Population[i].gene +Population[i].F*(Population[p_best].gene - Population[i].gene) + Population[i].F*(Population[p_r1].gene - x2.gene)
            vector_xnew = vector_xx
            #vector_xx = Population[p[0]].gene + F * (Population[p[1]].gene - Population[p[2]].gene)
            #vector_xnew = vector_xx
            #CrossOver Operation
            jrand = np.random.randint(Individual.DIMENSION)
            
            for j in range(Individual.DIMENSION):
                if (np.random.rand() >  1.0-Population[i].CR and not j==jrand):
                    vector_xnew[0,j] = Population[i].gene[0,j]
                if (vector_xnew[0,j] < MIN_VALUE or vector_xnew[0,j] > MAX_VALUE):
                    vector_xnew[0,j] = Population[i].gene[0,j]
            #Select Operation
            new_Individual = Individual.Indivi(batch_xs, batch_ys, vector_xnew)
            fitness = Individual.Indivi.Evaluate(Population[i],batch_xs, batch_ys)
            if (new_Individual.fitness_entropy < fitness):
                self.A.append(Population[i])
                self.S_CR.append(Population[i].CR)
                self.S_F.append(Population[i].F)
                Population[i] = new_Individual
                update += 1
            #Randomly remove solutions from A so that |A| <= Population size
            random.shuffle(self.A)
            while(len(self.A)>self.POPULATION):
                self.A.pop()
            # GeneX.append(Population[i].gene[0][0])
            # GeneY.append(Population[i].gene[0][1])
            # Fitness.append(Population[i].fitness)
        #print i,Population[i].F
        self.u_CR=(1.0-self.c)*self.u_CR + self.c*np.average(np.array(self.S_CR))
        self.u_F=(1.0-self.c)*self.u_F + self.c*np.average(np.array(self.S_F))
        
        # if (i==0):
            # print(self.u_CR)
          
        print("UpdatedIndividual=",update)
        UpdatedIndividual.append(update)


    def Average_Fitness(self):
        x =0.0
        for i in range(POPULATION):
            x += Population[i-1].fitness_entropy
        return x/POPULATION

    def Best_Individual(self):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        print( "Best Individual")
        print( Population[0].fitness_entropy ,Population[0].fitness_accuracy , Population[0].gene)

    def Best_plot(self,Bestindividual,Bestgene,Bestaccuracy):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        #Population.sort(key = operator.attrgetter('fitness'),reverse = True)
        Bestindividual.append(Population[0].fitness_entropy)
        # Bestgene.append(Population[0].gene)
        Bestaccuracy.append(Individual.Indivi.Evaluate2(Population[0],mnist.test.images,mnist.test.labels))
