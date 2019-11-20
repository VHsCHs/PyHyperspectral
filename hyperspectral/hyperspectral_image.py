import pandas as pd
import numpy as np
import torch
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import seaborn as sns

weights = [0.04,0.08,0.12,0.16,0.20,0.16,0.12,0.08,0.04]
weights = np.array(weights)
csv_file = "D:\\Desktop\\317.csv"
csv_data = pd.read_csv(csv_file,index_col = 0)
# csv_data = pd.read_csv(csv_file)
X = np.array(csv_data.index)
Y = np.array(csv_data)
Y = Y.T
# Y = np.sqrt(Y)
Y = 1 / Y
Y = np.log(Y)
Y_array = []
for line in Y:
    temp_array = []
    for index,sub in enumerate(line):
        if index-4 < 0:
            # average_array = []
            # for i in range(-4,5,1):
            #     if index+i-1 < 0:
            #         temp = line[index]
            #         average_array.append(temp)
            #     else:
            #         average_array.append(line[index+i+1])
            # average_array = np.array(average_array)
            temp_array.append(line[index])
            continue
        elif index + 5 > np.shape(line)[0]:
            # average_array = []
            # for i in range(-4, 5, 1):
            #     if index + i + 1 > np.shape(line)[0]:
            #         temp = line[index]
            #         average_array.append(temp)
            #     else:
            #         average_array.append(line[index + i])
            # average_array = np.array(average_array)
            temp_array.append(line[index])
            continue
        else:
            average_array = np.array(line[index - 4:index + 5])
        average = np.multiply(average_array,weights)
        average = np.sum(average,axis=0)
        temp_array.append(average)
    Y_array.append(temp_array)
Y = np.array(Y_array)
'''
微分变化
'''
Y_array = []
for i in range(6):
    temp_array = Y[i]
    for j in range(np.shape(temp_array)[0]):
        if j == np.shape(temp_array)[0]-1:
            continue
        else:
            temp_array[j] = temp_array[j+1] - temp_array[j]
    Y_array.append(temp_array)
Y = np.array(Y_array)
Y = np.divide(Y,X)

# scaler = StandardScaler()
# scaler.fit(Y)
# Y = scaler.transform(Y)
# print(csv_data['wave'])
# print(csv_data.iloc[:,1:-1])
for i in range(6):
    plt.plot(X,Y[i],linewidth=0.5)
plt.xlabel('waveband /nm')
plt.ylabel('abs /A')
plt.title('Hyperspectral Image')
plt.grid(c='b',ls='--',lw=0.5,fillstyle='full',alpha=0.3)
plt.axis([400,1000,-0.001,0.001])
plt.show()
