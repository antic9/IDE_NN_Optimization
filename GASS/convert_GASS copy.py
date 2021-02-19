import csv
import random

import numpy as np
import pandas as pd
import scipy.stats


class convert_GASS:
    def Convert():
        GASS_all_list = []
        GASS_input_all = []
        GASS_output_all = []
        GASS_input = []
        GASS_output = []
        GASS_input_test = []
        GASS_output_test = []
        testbatch_input = []
        testbatch_output = []
        dat = "GASS_data/Dataset/batch1.dat"
        df = pd.read_table(dat,sep="\s+",index_col=0,header=None)
        for c in df.columns.values:
            df[c] = df[c].apply(lambda x: float(str(x).split(':')[1]))
        dfs = df
        # for i in range(2,4):
        #     print(i)
        #     dat2 = "GASS_data/Dataset/batch"+str(i)+".dat"
        #     df2 = pd.read_table(dat2,sep="\s+",index_col=0,header=None)
        #     for c in df2.columns.values:
        #         df2[c] = df2[c].apply(lambda x: float(str(x).split(':')[1]))    
        #     dfs = dfs.append(df2)
        # dat2 = "GASS_data/Dataset/batch4.dat"
        # dftest = pd.read_table(dat2,sep="\s+",index_col=0,header=None)
        # for c in df2.columns.values:
        #     dftest[c] = dftest[c].apply(lambda x: float(str(x).split(':')[1]))   
        # for i in range(5,11):
        #     print(i)
        #     dat2 = "GASS_data/Dataset/batch"+str(i)+".dat"
        #     df2 = pd.read_table(dat2,sep="\s+",index_col=0,header=None)
        #     for c in df2.columns.values:
        #         df2[c] = df2[c].apply(lambda x: float(str(x).split(':')[1]))    
        #     dfs = dfs.append(df2)
        for i in range(2,11):
            print(i)
            dat2 = "GASS_data/Dataset/batch"+str(i)+".dat"
            df2 = pd.read_table(dat2,sep="\s+",index_col=0,header=None)
            for c in df2.columns.values:
                df2[c] = df2[c].apply(lambda x: float(str(x).split(':')[1]))    
            dfs = dfs.append(df2)
        GASS_output_all = dfs.index.values
        GASS_input_all = dfs.values
        # GASS_input_test = dftest.values
        # GASS_output_test = dftest.index.values
        for count in range(len(GASS_input_all)):
            temp = []
            temp.append(GASS_input_all[count])
            temp.append(GASS_output_all[count])
            GASS_all_list.append(temp)
        GASS_all = np.array(GASS_all_list)
        np.random.shuffle(GASS_all)
        for t3 in range(len(GASS_all)):
            if t3 < len(GASS_all)*0.8:
                GASS_input.append(GASS_all[t3][0])
                answer = GASS_all[t3][1]
                _output = np.zeros(6)
                _output[answer-1] = 1
                GASS_output.append(_output)
            else:
                testbatch_input.append(GASS_all[t3][0])
                answer = GASS_all[t3][1]
                _output = np.zeros(6)
                _output[answer-1] = 1
                testbatch_output.append(_output)
        GASS_input_test = testbatch_input
        GASS_output_test = testbatch_output
        # print(GASS_input)
        # print(GASS_output)
        # print(GASS_input_test)
        # print(GASS_output_test)
        # exit(0)
        return GASS_input,GASS_output,GASS_input_test,GASS_output_test
