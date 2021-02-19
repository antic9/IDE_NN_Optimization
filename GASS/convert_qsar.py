import numpy as np
import scipy.stats
import csv
import random

class convert_QSAR:
    def Convert():
        QSAR_data= []
        QSAR_input = []
        QSAR_output = []
        QSAR_output_test = []
        QSAR_input_test = []
        with open('QSAR_data/biodeg.csv') as f:
                reader = csv.reader(f)
                l = [row for row in reader]
        for t1 in range(len(l)):
            temp = l[t1][0]
            temp2 = np.array(temp.split(";"))
            QSAR_data.append(temp2)
        random.shuffle(QSAR_data)
        QSAR_data = np.array(QSAR_data)
        for t2 in range(len(QSAR_data)):
            temp_arr = []
            # print(temp_arr)
            temp_arr2 = []
            for t3 in range(len(QSAR_data[0])):
                if t2 <(len(QSAR_data)*0.8):
                    if(t3!=(len(QSAR_data[0])-1)):
                        # temp_arr=np.append(temp_arr,QSAR_data[t2][t3])
                        temp_arr.append(QSAR_data[t2][t3])

                    else:
                        if(QSAR_data[t2][t3]=="RB"):
                            QSAR_output.append([1,0])
                            # QSAR_output=np.append(QSAR_output,[1,0])
                        else:
                            QSAR_output.append([0,1])
                else:
                    if(t3!=(len(QSAR_data[0])-1)):
                        temp_arr.append(QSAR_data[t2][t3])
                    else:
                        if(QSAR_data[t2][t3]=="RB"):
                            QSAR_output_test.append([1,0])
                        else:
                            QSAR_output_test.append([0,1])
            temp_arr = np.array(temp_arr)
            temp_arr2 = temp_arr.astype(np.float)
            # print(temp_arr)
            if t2 <(len(QSAR_data)*0.8):
                # print(t2)
                QSAR_input.append(temp_arr2)
            else:
                QSAR_input_test.append(temp_arr2)
        QSAR_input = np.array(QSAR_input).astype(np.float)
        QSAR_input_test = np.array(QSAR_input_test).astype(np.float)
        QSAR_output = np.array(QSAR_output).astype(np.float)
        QSAR_output_test =np.array(QSAR_output_test).astype(np.float)
        return QSAR_input,QSAR_output,QSAR_input_test,QSAR_output_test