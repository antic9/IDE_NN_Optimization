# coding:UTF-8
import Individual
import numpy as np
import operator
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

import tensorflow as tf


#parameters
#GENERATION =  50
POPULATION =  0

MAX_VALUE = Individual.MAX_VALUE
MIN_VALUE = Individual.MIN_VALUE
gamma1 = 0.1
gamma2 = 0.1
F_l        =  0.1
F_u        =  0.9

tf.reset_default_graph()
Population = []

        
class adaptive_DE:


    def __init__(self,batch_xs, batch_ys,popu):
        global POPULATION 
        POPULATION = popu
        for p in range(POPULATION):
            Population.append(Individual.Indivi(batch_xs, batch_ys))
        

        #data strage
        #history_F = []
        #history_CR = []

    def Mutation(self,count, ps2, UpdatedIndividual,batch_xs, batch_ys):
        #Stack_NewIndividual for Scatter plot
        GeneX = []
        GeneY = []
        fitness_entropy = []
        fitness_accuracy = []
        update = 0
        for i in range(POPULATION):
            p = np.arange(POPULATION);p
            #p = np.delete(p,i)
            np.random.shuffle(p)
            # jDE Algorithm. Decide parameters F and CR
            if (np.random.rand() < gamma1):
                Population[i].F  = F_l + np.random.rand()*F_u

            if (np.random.rand() < gamma2):
                Population[i].CR = np.random.rand()


            vector_xx = Population[p[0]].gene + Population[i].F * (Population[p[1]].gene - Population[p[2]].gene)
            vector_xnew = vector_xx

            #Crossover Operation
            jrand = np.random.randint(Individual.DIMENSION)
            for j in range(Individual.DIMENSION):
                if (np.random.rand() >  1.0-Population[i].CR and not j==jrand):
                    vector_xnew[0,j] = Population[p[i]].gene[0,j]
                if (vector_xnew[0,j] < MIN_VALUE or vector_xnew[0,j] > MAX_VALUE):
                    vector_xnew[0,j] = Population[p[i]].gene[0,j]
            new_Individual = Individual.Indivi(batch_xs, batch_ys, vector_xnew)

            #Set new parameters F and CR
            new_Individual.F = Population[i].F
            new_Individual.CR = Population[i].CR

            #Select Operation
            fitness = Individual.Indivi.Evaluate(Population[i],batch_xs, batch_ys)
            if (new_Individual.fitness_entropy < fitness):
                Population[i] = new_Individual
                update += 1
            GeneX.append(Population[i].gene[0][0])
            GeneY.append(Population[i].gene[0][1])
            fitness_entropy.append(Population[i].fitness_accuracy)
            fitness_accuracy.append(Population[i].fitness_accuracy)
        print("UpdatedIndividual=",update)
        UpdatedIndividual.append(update)
        


        #Plot_Gene
        """
        if (Individual.DIMENSION == 2 ):
            x = GeneX
            y = GeneY
            z = Fitness
            fig = plt.figure()
            ax = fig.add_subplot(111, projection="3d")
            plt.hold(True);
            ax.scatter(x,y,z)
            plt.xlabel('x1')
            plt.ylabel('x2')
            ax.set_xlim([MIN_VALUE,MAX_VALUE])
            ax.set_ylim([MIN_VALUE,MAX_VALUE])
            ax.set_zlim(0,100)
            plt.savefig('Results/Scatter/jDE/Population_gene_Generation%s_Population%s.png' % (count, POPULATION-1))
            #history_CR[i].append(Population[i].CR)
            #history_F[i].append(Population[i].F)


            for n in len(history_CR):
                plt.plot(history_CR[n],'-o')
            plt.xlabel("Generation",size = 16)
            plt.ylabel("CR",size = 16)
            plt.show()

            for o in len(history_F):
                plt.plot(history_F[n],'-o')
            plt.xlabel("Generation",size = 16)
            plt.ylabel("F",size = 16)
            plt.show()
        """

    def Average_Fitness(self):
        x =0.0
        for i in range(POPULATION):
            
            x += Population[i-1].fitness_accuracy
        return x/POPULATION

    def Best_Individual(self):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        #Population.sort(key = operator.attrgetter('fitness'),reverse = True)
        print( "Best Individual")
        print( Population[0].fitness_entropy ,Population[0].fitness_accuracy , Population[0].gene)

    def Best_plot(self,Bestindividual,Bestgene,Bestaccuracy,test_input,test_output):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        #Population.sort(key = operator.attrgetter('fitness'),reverse = True)
        Bestindividual.append(Population[0].fitness_entropy)
        Bestgene.append(Population[0].gene)
        Bestaccuracy.append(Individual.Indivi.Evaluate2(Population[0],test_input,test_output))
        

