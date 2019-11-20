import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import math
import time
from sklearn.decomposition import PCA
from scipy.spatial.distance import pdist
from sklearn.feature_selection import SelectKBest,SelectPercentile,f_regression,mutual_info_regression
from sklearn.covariance import EllipticEnvelope
start_time = time.time()
'''
pandas导入csv
'''
input_location = 'D:\\Desktop\\dataset\\hyperspectral-soilmoisture-dataset-v1.0.0\\felixriese-hyperspectral-soilmoisture-dataset-657d5d0\\soilmoisture_dataset.csv'
csv_data = pd.read_csv(filepath_or_buffer=input_location,
                       sep=',',
                       na_values='NULL',
                       header=0,
                       index_col=[0,1]
                       )

df = pd.DataFrame(csv_data)
X_array = np.array(df.iloc[:,2:df.shape[1]])
# print(X_array)
X_label = np.array(df.columns)
X_label = X_label[1:X_label.shape[0]]
X_label = np.array(X_label)

Y_array = np.array(df.loc[:,'soil_moisture'])
# Y_array = Y_array.reshape(-1,1)
Y_array = np.array(Y_array)

'''
离群值检测
'''
cov = EllipticEnvelope(random_state=1,contamination=0.05)
cov.fit(np.hstack([X_array,Y_array.reshape(-1,1)]))
index = cov.predict(np.hstack([X_array,Y_array.reshape(-1,1)]))

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

print(all_np(index))

def error_wipe(array,index):
    temp_array = []
    for num,line in enumerate(array):
        if index[num] == 1:
            temp_array.append(line)
        elif index[num] == -1:
            # print(line)
            pass
    temp_array = np.array(temp_array)
    return temp_array

X_array = error_wipe(X_array,index)
Y_array = error_wipe(Y_array,index)

# '''
# 异常样本剔除
# '''
# X = np.hstack([X_array,Y_array.reshape(-1,1)])
# XT = X
# X = X.T
# #方法一：根据公式求解
# S=np.cov(X)   #两个维度之间协方差矩阵
# SI = np.linalg.inv(S) #协方差矩阵的逆矩阵
# #马氏距离计算两个样本之间的距离，此处共有10个样本，两两组合，共有45个距离。
# MD=[]
# d=[]
# delta = X - np.mean(X,axis=0)
# for i in delta.T:
#     d=np.sqrt(np.dot(np.dot(i,SI),i.T))
#     MD.append(d)
# MD = np.array(MD)
# print(MD)
#
# #方法二：根据scipy库求解
# d2=pdist(XT,'mahalanobis')
# print(d2)

'''
以标准化的含水量作为透明度
'''
def standard_zero_to_one(array):
    mean = array.mean()         #计算平均数
    deviation = array.std()     #计算标准差
    # 标准化数据的公式: (数据值 - 平均数) / 标准差
    standard = np.array((array - mean) / deviation)
    standard = standard + np.abs(standard.min())
    standard = standard / standard.max()
    return standard
alpha = standard_zero_to_one(Y_array)

'''
样本划分
'''
X_array,X_test,Y_array,Y_test = train_test_split(X_array, Y_array, train_size=500, random_state=1)
# print(np.shape(X_test))
# X_array,X_test,Y_array,Y_test =

'''
卡方检验
'''
# X_best = SelectKBest(f_regression, k=10).fit_transform(X_array, Y_array)
X_transform = SelectKBest(f_regression,k='all')
X_transform.fit(X_array,Y_array)
X_array = X_transform.transform(X_array)
X_test = X_transform.transform(X_test)
# print(X_transform.get_support())
# print(X_transform.scores_)
# print(np.shape(X_array))
# print(np.shape(X_test))

'''
标准化
'''
scaler = StandardScaler()
scaler.fit(X_array)
X_array = scaler.transform(X_array)
X_test = scaler.fit_transform(X_test)

'''
#高光谱作图
for n,i in enumerate(X_array):
    Y = i[1:i.shape[0]]
    X = X_label[1:X_label.shape[0]]
    plt.plot(X,Y,linewidth=0.5,color='r',alpha=Y_Alpha[n])

plt.xlabel('waveband /nm')
plt.ylabel('abs /A')
plt.title('Hyperspectral Image')
plt.tick_params(axis='x', rotation=70)
plt.grid(c='b',ls='--',lw=0.5,fillstyle='full',alpha=0.3)
# plt.axis([400,1000,-0.001,0.001])
plt.show()
'''

# '''
# PCA降维
# '''
# pca = PCA(n_components='mle',svd_solver='full')
# # pca = PCA(n_components=9)
# X_array = pca.fit_transform(X_array)
# X_test = pca.transform(X_test)

'''
多层感知机初始化
'''
clf = MLPRegressor(
    hidden_layer_sizes=(500,30), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000, shuffle=True,
    random_state=1, tol=1e-4, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf.fit(X_array, Y_array)
Y_predict = clf.predict(X_test)
print('R^2=', clf.score(X_test, Y_test))

def RMSE(origin_data,predict_data):
    '''
    均方根误差
    √[∑di^2/n]=Re
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.array(origin_data)
    origin_data = origin_data.reshape(-1,1)
    predict_data = np.array(predict_data)
    predict_data = predict_data.reshape(-1,1)
    temp = origin_data - predict_data
    temp = np.multiply(temp,temp)
    total = np.shape(temp)[0]
    total = float(total)
    temp = float(np.sum(temp))
    rmse = math.sqrt(temp / total)
    return rmse

print('RMSE=', RMSE(Y_test, Y_predict))

'''
plt作图
'''

# coefs = clf.coefs_
# for i in coefs:
#     print(i.shape)

end_time = time.time()
total_time = end_time - start_time
print('total time = ' + str(total_time) + 's')

plt_origin = plt.scatter(X_test[:, 0], Y_test)
plt_predict = plt.scatter(X_test[:, 0], Y_predict)
plt.xlabel('abs /A')
plt.ylabel('moisture /%')
plt.legend(handles=[plt_origin,plt_predict],labels=['origin','predict'],loc='best')
plt.yticks([25,30,35,40,45,50],['min','30%','35%','40%','45%','max'])
plt.title('soilmoisture_dataset')
plt.axis([-2,2,25,50])
plt.show()

