import csv
import random

import numpy as np
import pandas as pd
import scipy.stats


class convert_MAGIC:
    def Convert():
        MAGIC_all_list = []
        MAGIC_output_all = []
        MAGIC_input = []
        MAGIC_output = []
        MAGIC_input_test = []
        MAGIC_output_test = []
        dat = "MAGIC_data/magic04.data"
        df = pd.read_table(dat,sep="\s+",index_col=0,header=None)
        for c in df.columns.values:
            df[c] = df[c].apply(lambda x: float(str(x).split(',')[1])) 
        MAGIC_output_all = df.index.values
        # print(MAGIC_output_all)
        for count in range(len(MAGIC_output_all)):
            temp = np.array(MAGIC_output_all[count].split(","))
            MAGIC_all_list.append(temp)
        MAGIC_all = np.array(MAGIC_all_list)
        np.random.shuffle(MAGIC_all)
        MAGIC_all = np.array(MAGIC_all)
        for t2 in range(len(MAGIC_all)):
            temp_arr = []
            # print(temp_arr)
            temp_arr2 = []
            for t3 in range(len(MAGIC_all[0])):
                if t2 <(len(MAGIC_all)*0.8):
                    if(t3!=(len(MAGIC_all[0])-1)):
                        # temp_arr=np.append(temp_arr,MAGIC_all[t2][t3])
                        temp_arr.append(MAGIC_all[t2][t3])

                    else:
                        if(MAGIC_all[t2][t3]=="g"):
                            MAGIC_output.append([1,0])
                            # MAGIC_output=np.append(MAGIC_output,[1,0])
                        else:
                            MAGIC_output.append([0,1])
                else:
                    if(t3!=(len(MAGIC_all[0])-1)):
                        temp_arr.append(MAGIC_all[t2][t3])
                    else:
                        if(MAGIC_all[t2][t3]=="g"):
                            MAGIC_output_test.append([1,0])
                        else:
                            MAGIC_output_test.append([0,1])
            temp_arr = np.array(temp_arr)
            temp_arr2 = temp_arr.astype(np.float)
            # print(temp_arr)
            if t2 <(len(MAGIC_all)*0.8):
                # print(t2)
                MAGIC_input.append(temp_arr2)
            else:
                MAGIC_input_test.append(temp_arr2)
        MAGIC_input = np.array(MAGIC_input).astype(np.float)
        MAGIC_input_test = np.array(MAGIC_input_test).astype(np.float)
        MAGIC_output = np.array(MAGIC_output).astype(np.float)
        MAGIC_output_test =np.array(MAGIC_output_test).astype(np.float)
        # print(len(MAGIC_all_list))
        # print(MAGIC_all.shape)
        # print(MAGIC_input.shape)
        # print(MAGIC_output.shape)
        # print(MAGIC_input_test.shape)
        # print(MAGIC_output_test.shape)
        # print(MAGIC_all_list)
        # print(MAGIC_all)
        # print(MAGIC_input[0][0])
        # print(MAGIC_output)
        # print(MAGIC_input_test)
        # print(MAGIC_output_test)
        # exit(0)
        return MAGIC_input,MAGIC_output,MAGIC_input_test,MAGIC_output_test
