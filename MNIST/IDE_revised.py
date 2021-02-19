# coding:UTF-8
import Indi as Individual
import numpy as np
import operator
import random
import main
import matplotlib
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
#from dpgmm2 import DPGMM
matplotlib.use('Agg')
import matplotlib.pyplot as plt


#parameters
POPULATION =  50
MAX_VALUE = Individual.MAX_VALUE
MIN_VALUE = Individual.MIN_VALUE
F_l        =  0.1
F_u        =  0.9
#CR         =  0.5
gamma1     =  0.1
gamma2     =  0.1
eps = 1e-3
Population = []
SPACE = MAX_VALUE - MIN_VALUE
class IDE_NN:
    def __init__(self,batch_xs, batch_ys, u_CR=0.5,u_F=0.5,p2=0.05,c=0.1,POPULATION=POPULATION):
        
        zero_dim=[]
        for k in range(batch_xs.shape[1]):
                z_count = 0
                for j in range(batch_xs.shape[0]):
                    if batch_xs[j,k] == 0:
                        z_count += 1
                if z_count == batch_xs.shape[0]:
                    zero_dim.append(k)
        for pop in range(POPULATION):
            Population.append(Individual.Indivi(batch_xs, batch_ys,zero_dim=zero_dim))
            self.u_CR=u_CR
            self.u_F=u_F
            self.p2=p2
            self.c=c
            self.POPULATION=POPULATION
            self.A=[]
            self.S_CR=[]
            self.S_F=[]

    """
    def init(self):
        for pop in xrange(POPULATION):
            Population.append(Individual.Individual())
    """
    def Mutation(self,count,ps2,UpdatedIndividual,batch_xs, batch_ys):
        #print("poprange=",len(Population))
        #Stack_NewIndividual for Scatter plot
        GeneX = []
        GeneY = []
        fitness_entropy = []
        fitness_accuracy = []
        update = 0
        zero_dim=[]
        for k in range(batch_xs.shape[1]):
                z_count = 0
                for j in range(batch_xs.shape[0]):
                    if batch_xs[j,k] == 0:
                        z_count += 1
                if z_count == batch_xs.shape[0]:
                    zero_dim.append(k)
        #Population.sort(key = operator.attrgetter('fitness'),reverse = False)
        c = count
        float (c)
        gcount = c / main.GENERATION
        float (gcount)
        #print "p_best=",p_best
        #Set Ps Pd and dr3 for mutant vector
        ps = 0.1 + 0.9*10 ** (5 * (gcount-1))
        #print("ps=",ps)
        #pd = ps2 * 0.1
        #print("pd=",pd)
        #Srange = int(POPULATION*ps)
        #Irange = POPULATION - Srange
        #print("Srange=",Srange)
        #print("Irange=",Irange)
        #pdlist.append(pd)
        #print "ps2=",ps2
        #Mutation
        for i in range(POPULATION):
            #print "population",i,"ps2=",Population[i].ps2
            #Make P_best
            #Population.sort(key = operator.attrgetter('fitness'),reverse = False)

            #ps2 = Population[i].ps2
            ps2 = ps
           
            #print "pop",i,"ps2=",ps2
            pd = ps2 * 0.1
            #print "Population",i,"ps2=",ps2
            Srange = int(POPULATION*ps2)
            #print(Srange)
                        
            Irange = POPULATION - Srange
            p = np.arange(POPULATION);p
            #p = np.delete(p,i)
            
            np.random.shuffle(p)
            if(count < 10 ): item = i
            else:
                item = p[0]
                
            #print("population[%s]=" % i ,Population[i].fitness)
            #Classify two sets and Create Xbetter
            As = []
            for l in range (len(p)):
                As.append(Population[p[l]].fitness_entropy)
            As.sort()
            #print("Asfitness=",As)
            Ss = []
            for m in range (0,Srange):
                Ss.append(As[m])
            #print("Ss=",Ss)
            Is = []
            for n in range (Srange,len(As)):
                Is.append(As[n])
            #print("Is=",len(Is))
            Ss_gene = []
            for o in range (POPULATION):
                if Population[p[o]].fitness_entropy in Ss:
                    xbetter = Population[p[o]].fitness_entropy
                    Ss_gene.append(Population[p[o]].gene)
            Xb = random.choice(Ss_gene)

            #print("xbetter=",xbetter)
            #print("xb=",x2.gene)
            #Create MutantVector

            dr3 = np.empty((1,Individual.DIMENSION))
            for k in range (Individual.DIMENSION):
                if np.random.rand() < pd:
                    dr3[0,k] =  MIN_VALUE + np.random.rand()*( MAX_VALUE - MIN_VALUE)
                else:
                    dr3[0,k] = Population[p[3]].gene[0,k]

            #dr3 = Population[p[3]].gene
            #print("populationp3=",Population[p[3]].gene)
            #print("dr3=",dr3)
            fNP = float(i)/float(len(Population))
            F = np.random.normal(fNP,0.1)
            if Population[i].fitness_entropy in Ss:
                vector_xx = Population[item].gene + F  * (Population[p[1]].gene - Population[item].gene) + F * (Population[p[2]].gene - dr3)
                vector_xnew = vector_xx
                #print("Ss")
            elif Population[i].fitness_entropy in Is:
                vector_xx = Population[item].gene + F * (Xb - Population[item].gene) + F * (Population[p[2]].gene - dr3)
                #vector_xx = F * (p_best - Population[i].gene) + F *  (Population[p[1]].gene- x2.gene)
                vector_xnew = vector_xx
                #print("Is")
            else:
                print("No match")
                break
            #CrossOver Operation
            jrand = np.random.randint(Individual.DIMENSION)
            cNP = float(i)/float(len(Population))
            tCR=np.random.normal(cNP,0.1)
            for j in range(Individual.DIMENSION):
                if ( np.random.rand() > tCR and not j==jrand):
                    vector_xnew[0,j] = Population[i].gene[0,j]
                if (vector_xnew[0,j] < MIN_VALUE or vector_xnew[0,j] > MAX_VALUE):
                    vector_xnew[0,j] = MIN_VALUE + np.random.rand()*( MAX_VALUE - MIN_VALUE)
            #print("vectorxnew=",vector_xnew)
                    #vector_xnew[0,j] = MIN_VALUE + np.random.rand() * (MAX_VALUE - MIN_VALUE)
            #Select Operation
            new_Individual = Individual.Indivi(batch_xs, batch_ys, gene = vector_xnew)
            fitness = Individual.Indivi.Evaluate(Population[i],batch_xs,batch_ys,zero_dim)
            if (new_Individual.fitness_entropy < fitness):
                Population[i] = new_Individual
                update += 1
            GeneX.append(Population[i].gene[0][0])
            GeneY.append(Population[i].gene[0][1])
            fitness_entropy.append(Population[i].fitness_entropy)
            fitness_accuracy.append(Population[i].fitness_accuracy)
        print("UpdatedIndividual=",update)
        #print "Bestindiv=",Population[30].gene
        UpdatedIndividual.append(update)
        #print "dim1",Population[i].gene[0][0]
        #print "dim2",Population[i].gene[0][1]
        #print "dim3",Population[i].gene[0][2]
        
        
    def euclid(self,euclidlist,euclidgraph,euclidavelist):
        varlist = []
        euclidlist2 = []
        for i in range(POPULATION):
            varlist.append(Population[i].gene)
        varar = np.array(varlist)
        gravity=np.mean(varar, axis=0)

        for j in range(POPULATION):
            euclidtemp =0.0
            for k in range (Individual.DIMENSION):
                euclidtemp += (varar[j,0,k] - gravity[0,k])**2
            euclid = np.sqrt(euclidtemp)
            euclidlist2.append(euclid)
        euclidlist3 = np.array(euclidlist2)
        #print("eurange=",len(euclidlist3))
        euclidlist.append(euclidlist3)
        euclidgraph.append(euclidlist3)
        euclidave = np.average(euclidlist3)
        
        for j in range(POPULATION):
            temp = (euclidlist3-euclidave)**2
        temp = temp/len(Population)
        euclidave = np.average(temp)
        euclidavelist.append(euclidave)
        
        

    
    def Average_Fitness(self):
        x =0.0
        for i in range(POPULATION):
            #print("Population.fitness=",Population[i].fitness)
            x += Population[i].fitness_entropy
        return x/POPULATION


    def Best_Individual(self):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        #Population.sort(key = operator.attrgetter('fitness'),reverse = True)
        print( "Best Individual")
        # print( Population[0].fitness_entropy ,Population[0].fitness_accuracy,Population[0].gene)

        print( Population[0].fitness_entropy ,Population[0].fitness_accuracy)

    def Best_plot(self,Bestindividual,Bestgene,Bestaccuracy):
        Population.sort(key = operator.attrgetter('fitness_entropy'),reverse = False)
        #Population.sort(key = operator.attrgetter('fitness'),reverse = True)
        Bestindividual.append(Population[0].fitness_entropy)
        Bestgene.append(Population[0].gene)
        Bestaccuracy.append(Individual.Indivi.Evaluate2(Population[0],mnist.test.images,mnist.test.labels))
        
