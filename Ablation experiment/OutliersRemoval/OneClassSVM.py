import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
import math
import time
from sklearn.svm import OneClassSVM
from private_model.pymodel import RMSE

start_time = time.time()
'''
pandas导入csv
'''
input_location = 'D:\\Desktop\\dataset\\hyperspectral-soilmoisture-dataset-v1.0.0\\felixriese-hyperspectral-soilmoisture-dataset-657d5d0\\soilmoisture_dataset.csv'
csv_data = pd.read_csv(filepath_or_buffer=input_location,
                       sep=',',
                       na_values='NULL',
                       header=0,
                       index_col=[0, 1]
                       )

df = pd.DataFrame(csv_data)
X_array = np.array(df.iloc[:, 2:df.shape[1]])
X_label = np.array(df.columns)
X_label = X_label[1:X_label.shape[0]]
X_label = np.array(X_label)

Y_array = np.array(df.loc[:, 'soil_moisture'])
Y_array = np.array(Y_array)

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


# print(index)
# print(cov.location_)
# print(cov.covariance_)
# print(cov.precision_)
# print(cov.support_)
# print(cov.offset_)

def all_np(arr):
    arr = np.array(arr)
    key = np.unique(arr)
    result = {}
    for k in key:
        mask = (arr == k)
        arr_new = arr[mask]
        v = arr_new.size
        result[k] = v
    return result


def error_wipe(array, index):
    valid = []
    invalid = []
    for num, line in enumerate(array):
        if index[num] == 1:
            valid.append(line)
        elif index[num] == -1:
            invalid.append(line)
    valid = np.array(valid)
    invalid = np.array(invalid)
    return valid, invalid


def floatrange(start, stop, step):
    steps = math.floor((stop - start) / step)
    temp = []
    for i in range(steps):
        temp.append(start + step * i)
    return temp


# '''
# 离群值检测
# '''
# plt.figure('Isolation Forest', figsize=(9.6, 3.8), dpi=200)
# # plt.suptitle('Elliptic Envelope contamination', fontsize=20)
# for num, i in enumerate(list([0.01, 0.03, 0.05, 0.1, 0.2, 0.5])):
#     one = OneClassSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=i, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=1)
#     one.fit(X_array)
#     index = one.predict(X_array)
#
#     X_valid, X_invalid = error_wipe(X_array, index)
#     Y_valid, Y_invalid = error_wipe(Y_array, index)
#
#     reg1 = LinearRegression()
#     reg1.fit(X_valid, Y_valid)
#     reg2 = LinearRegression()
#     reg2.fit(X_array, Y_array)
#     reg1.rmse = RMSE(Y_valid, reg1.predict(X_valid))
#     reg2.rmse = RMSE(Y_array, reg2.predict(X_array))
#     print('reg1', reg1.score(X_valid, Y_valid))
#     print('reg2', reg2.score(X_array, Y_array))
#
#     print(all_np(index))
#     print(random.sample(range(100), 10))
#     plt.subplot(2, 3, num + 1)
#     plt.title('nu = ' + str(i), fontsize=10)
#     plt.scatter(X_valid[:, 48], Y_valid, color='chocolate', s=10)
#     plt.scatter(X_invalid[:, 48], Y_invalid, color='teal', s=15)
#     plt.xticks([]), plt.yticks(fontsize=6)
#     plt.text(0.16, 38, '原始R^2 = ' + str(round(reg2.score(X_array, Y_array), 3)) + '\n去除异常值后R^2 = ' + str(
#         round(reg1.score(X_valid, Y_valid), 3)) + '\n原始RMSE=' + str(round(reg2.rmse, 3)) + '\n去除异常值后RMSE=' + str(
#         round(reg1.rmse, 3)), fontsize=6)
#
# plt.savefig('D:\Desktop\hyper\OneClassSVM contamination.jpeg', dpi=400)
# plt.show()


one = OneClassSVM(kernel='rbf', degree=3, gamma='auto', coef0=0.0, tol=0.001, nu=0.3, shrinking=True, cache_size=200, verbose=False, max_iter=-1, random_state=1)
one.fit(X_array)
index = one.predict(X_array)
print(all_np(index))

X_valid, X_invalid = error_wipe(X_array, index)
Y_valid, Y_invalid = error_wipe(Y_array, index)

reg = LinearRegression()
reg.fit(X_valid,Y_valid)
print(str(round(reg.score(X_valid,Y_valid),3)))
print(str(round(RMSE(Y_valid,reg.predict(X_valid)),3)))
plt.figure('OneClassSVM',figsize=(4.8,5.4),dpi=100)
plt.title('OneClassSVM',fontsize=20)
plt.scatter(X_array[:,48],Y_array,color='teal', s=10)
plt.scatter(X_valid[:,48],Y_valid,color='chocolate', s=15)
plt.savefig('D:\Desktop\hyper\OneClassSVM.jpg',dpi=100)
plt.show()