# -*- coding: utf-8 -*-
import csv
import sys

import matplotlib.pyplot as plt
import numpy as np
import scipy.stats
np.set_printoptions(threshold = np.inf)

MAGIC_data= []
MAGIC_input = []
MAGIC_output = []
MAGIC_output_test = []
MAGIC_input_test = []
# MAGIC_input = np.empty((844,41))
# MAGIC_output = np.empty((844,2))
# MAGIC_output_test = np.empty((211,41))
# MAGIC_input_test = np.empty((211,2))

with open('magic04.data') as f:
    reader = csv.reader(f)
    l = [row for row in reader]
for i in range(len(l)):
    temp = l[i][0]
    temp2 = np.array(temp.split(","))
    MAGIC_data.append(temp2)
MAGIC_data = np.array(MAGIC_data)
print(MAGIC_data)
exit(0)
for t2 in range(len(MAGIC_data)):
    temp_arr = []
    # print(temp_arr)
    temp_arr2 = []
    for t3 in range(len(MAGIC_data[0])):
        if t2 <(len(MAGIC_data)*0.8):
            if(t3!=(len(MAGIC_data[0])-1)):
                # temp_arr=np.append(temp_arr,MAGIC_data[t2][t3])
                temp_arr.append(MAGIC_data[t2][t3])

            else:
                if(MAGIC_data[t2][t3]=="g"):
                    MAGIC_output.append([1,0])
                    # MAGIC_output=np.append(MAGIC_output,[1,0])
                else:
                    MAGIC_output.append([0,1])
        else:
            if(t3!=(len(MAGIC_data[0])-1)):
                temp_arr.append(MAGIC_data[t2][t3])
            else:
                if(MAGIC_data[t2][t3]=="h"):
                    MAGIC_output_test.append([1,0])
                else:
                    MAGIC_output_test.append([0,1])
    temp_arr = np.array(temp_arr)
    temp_arr2 = temp_arr.astype(np.float)
    # print(temp_arr)
    if t2 <(len(MAGIC_data)*0.8):
        # print(t2)
        MAGIC_input.append(temp_arr2)
    else:
        MAGIC_input_test.append(temp_arr2)
MAGIC_input = np.array(MAGIC_input).astype(np.float)
MAGIC_input_test = np.array(MAGIC_input_test).astype(np.float)
MAGIC_output = np.array(MAGIC_output).astype(np.float)
MAGIC_output_test =np.array(MAGIC_output_test).astype(np.float)
print(type(MAGIC_input[0]))
print(MAGIC_input.shape)
# print(len(MAGIC_input))
print(MAGIC_input_test.shape)
print(MAGIC_output.shape)
print(MAGIC_output_test.shape)
# MAGIC_input = MAGIC_input.reshape((844,41))
# MAGIC_output = MAGIC_output.reshape((844,2))
# MAGIC_output_test = MAGIC_output_test.reshape((211,41))
# MAGIC_input_test = MAGIC_input_test.reshape((211,2))
print(MAGIC_input)
print(MAGIC_input_test)
print(MAGIC_output)
print(MAGIC_output_test)


            
            