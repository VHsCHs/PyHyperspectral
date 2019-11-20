#!/usr/bin/env python
# -*- coding: utf-8 -*-

from sklearn.decomposition import PCA
from sklearn.metrics import r2_score
from sklearn.preprocessing import StandardScaler, MinMaxScaler, MaxAbsScaler, Normalizer, Binarizer, RobustScaler, \
    PolynomialFeatures
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import IsolationForest
from models.pymodel import error_wipe, RMSE, all_np
import numpy as np
import pandas as pd
import time

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

file = 'D:\\Desktop\\dataset\\hyperspectral-soilmoisture-dataset-v1.0.0\\felixriese-hyperspectral-soilmoisture-dataset-657d5d0\\soilmoisture_dataset.csv'
with open(file, 'r') as f:
    data = pd.read_csv(file, index_col=[0], header=[0])
    x = data.loc[:, '454':'950']
    y = data.loc[:, 'soil_moisture']
    x = np.array(x)
    y = np.array(y)

OutliersRemover = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, max_features=1.0,
                                  bootstrap=False,
                                  n_jobs=1, random_state=1, verbose=0, behaviour='new')
OutliersRemover.fit(x)
index = OutliersRemover.predict(x)
print(all_np(index))
x = error_wipe(x, index)
y = error_wipe(y, index)
print(np.shape(x))
print(np.shape(y))
x_train, x_test, y_train, y_test = train_test_split(x, y, train_size=500, random_state=1)
x_label = x_test[:, 0]
method_name = ['Origin', 'MinMaxScaler', 'StdScaler', 'MaxAbsScaler', 'Normalization', 'Binarizer', 'RobustScaler',
               'PolynomialFeatures']
methods = [Pipeline([('MLR', MLPRegressor(
    hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000, shuffle=True,
    random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ]),
           Pipeline([('minmax_scaler', MinMaxScaler()),
                     ('MLR', MLPRegressor(
                         hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5,
                         max_iter=100000,
                         shuffle=True,
                         random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ]),

           Pipeline([('StdScaler', StandardScaler()),
                     ('MLR', MLPRegressor(
                         hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000,
                         shuffle=True,
                         random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ]),
           Pipeline([('MaxAbsScaler', MaxAbsScaler()),
                     ('MLR', MLPRegressor(
                         hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000,
                         shuffle=True,
                         random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ]),
           Pipeline([('Normalization', Normalizer(norm='l2')),
                     ('MLR', MLPRegressor(
                         hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000,
                         shuffle=True,
                         random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ]),
           Pipeline([('Binarizer', Binarizer()),
                     ('MLR', MLPRegressor(
                         hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000,
                         shuffle=True,
                         random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ]),
           Pipeline([('RobustScaler', RobustScaler()),
                     ('MLR', MLPRegressor(
                         hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000,
                         shuffle=True,
                         random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ]),
           Pipeline([('PCA', PCA(n_components=9)),
                     ('PolynomialFeatures', PolynomialFeatures(2)),
                     ('MLR', MLPRegressor(
                         hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000,
                         shuffle=True,
                         random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
                     ])
           ]
plt.figure('Normalize', figsize=(9.6, 5.4), dpi=100)


# plt.suptitle('Normalize Methods', fontsize=30)


def str_round(text, decimal=3):
    return str(round(text, decimal))


for num, method in enumerate(methods):
    start_time = time.time()
    method.fit(x_train, y_train)
    prediction = method.predict(x_test)
    r2 = str_round(r2_score(y_test, prediction), 3)
    rmse = str_round(RMSE(y_test, prediction), 3)

    plt.subplot(2, 4, num + 1)
    plt.title(method_name[num], fontsize=15)
    plt.scatter(x_label, y_test, c='chocolate', s=10, label='origin')
    plt.scatter(x_label, prediction, c='teal', s=3, label='predict')
    plt.text(0.13, 37, 'r^2=' + r2 + '\nrmse=' + rmse, fontsize=8,bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="gray", lw=1,alpha=0.8))
    plt.xticks([]), plt.yticks(fontsize=6)
    plt.legend(fontsize=7)

    print(method_name[num])
    # print(str_round(r2_score(y_test, prediction), 3))
    # print(str_round(RMSE(y_test, prediction), 3))
    end_time = time.time()
    tol_time = end_time - start_time
    print('total time=' + str(round(tol_time, 3)) + 's')
    plt.savefig('D:\\Desktop\\normalize\\normalize.jpg', dpi=400)
plt.show()
