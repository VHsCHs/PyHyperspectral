# -*- coding: utf-8 -*-
import numpy as np
import pandas as pd
import time
from matplotlib import pyplot as plt

from models.pymodel import RMSE
from sklearn.metrics import r2_score

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import EllipticEnvelope
from sklearn.linear_model import LinearRegression, LogisticRegression, SGDRegressor, Ridge, Lasso, RANSACRegressor, \
    BayesianRidge
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR, LinearSVR, NuSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.neighbors import KNeighborsRegressor, RadiusNeighborsRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, ExtraTreesRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.kernel_ridge import KernelRidge

start_time = time.time()

names = ['Linear Regression-最小二乘回归', 'PLS Regression-偏最小二乘回归', 'Logistic Regression-逻辑回归',
         'SGD Regressor-梯度下降法', 'PLS Canonical-经典偏最小二乘', 'CCA-典型关联分析',
         'MLP Regressor-多层感知机', 'SVR-支持向量机回归', 'Linear SVR-线性支持向量机',
         'Nu SVR-Nu支持向量机', 'Decision Tree-决策树', 'Gaussian Process-高斯过程',
         'K-Neighbors-K邻近算法', 'Radius Neighbors-限定半径最近邻回归',
         'Random Forest-随机森林', 'Ada Boost-自适应增强', 'Gradient Boosting-梯度上升',
         'Gaussian NB-高斯分布朴素贝叶斯', 'Ridge-岭回归', 'Lasso-套索回归',
         'Kernel Ridge-核岭回归', 'RANSAC Regressor-鲁棒回归算法',
         'Bayesian Regressor-贝叶斯回归', 'Extra Tree-极端随机森林']

random_state = 50
normalize = True
n_components = 50
tol = 1e-04

regressors = [
    LinearRegression(fit_intercept=True, normalize=normalize, copy_X=True, n_jobs=None),
    PLSRegression(n_components=n_components, scale=True, max_iter=500, tol=tol, copy=True),
    LogisticRegression(penalty='l2', dual=False, tol=tol, C=1.0, fit_intercept=True,
                       intercept_scaling=1, class_weight=None, random_state=random_state, solver='lbfgs',
                       max_iter=100000, multi_class='auto', verbose=0, warm_start=False, n_jobs=None, l1_ratio=None),
    SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                 tol=tol, shuffle=True, verbose=0, epsilon=0.1, random_state=random_state, learning_rate='invscaling',
                 eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                 warm_start=False, average=False),
    PLSCanonical(n_components=9, scale=False, algorithm='nipals', max_iter=1000, tol=1e-3, copy=True),
    CCA(n_components=9, scale=False, max_iter=1000, tol=1e-3, copy=True),
    MLPRegressor(hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto',
                 learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000, shuffle=True,
                 random_state=random_state, tol=tol, verbose=False, warm_start=False, momentum=0.9,
                 nesterovs_momentum=True, early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08),
    SVR(kernel='rbf', degree=3, gamma='scale', coef0=0.0, tol=tol, C=1.0, epsilon=0.1, shrinking=True,
        cache_size=200, verbose=False, max_iter=-1),
    LinearSVR(epsilon=0.0, tol=tol, C=1.0, loss='epsilon_insensitive', fit_intercept=True, intercept_scaling=1.0,
              dual=True, verbose=0, random_state=random_state, max_iter=1000000),
    NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale', coef0=0.0, shrinking=True, tol=tol,
          cache_size=200, verbose=False, max_iter=-1),
    DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None, min_samples_split=2, min_samples_leaf=1,
                          min_weight_fraction_leaf=0.0, max_features=None, random_state=random_state,
                          max_leaf_nodes=None, min_impurity_decrease=0.0, min_impurity_split=None, presort=False),
    GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
                             normalize_y=normalize, copy_X_train=True, random_state=random_state),
    KNeighborsRegressor(n_neighbors=10, weights='uniform', algorithm='auto', leaf_size=30, p=2, metric='minkowski',
                        metric_params=None, n_jobs=None),
    RadiusNeighborsRegressor(radius=5.0, weights='uniform', algorithm='kd_tree', leaf_size=50, p=2, metric='minkowski',
                             metric_params=None, n_jobs=None),
    RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=9,
                          min_samples_split=2, min_samples_leaf=1,
                          min_weight_fraction_leaf=0.0, max_features=1,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, bootstrap=True, oob_score=True,
                          n_jobs=None, random_state=random_state, verbose=0, warm_start=False),
    AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0, loss='linear',
                      random_state=random_state),
    GradientBoostingRegressor(loss='ls', learning_rate=0.1, n_estimators=100, subsample=1.0, criterion='friedman_mse',
                              min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0, max_depth=3,
                              min_impurity_decrease=0.0, min_impurity_split=None, init=None, random_state=random_state,
                              max_features=None, alpha=0.9, verbose=0, max_leaf_nodes=None, warm_start=False,
                              presort='auto', validation_fraction=0.1, n_iter_no_change=None, tol=tol),
    GaussianNB(priors=None, var_smoothing=1e-09),
    Ridge(alpha=1.0, fit_intercept=True, normalize=False, copy_X=True, max_iter=None, tol=tol,
          solver='auto', random_state=None),
    Lasso(alpha=0.5, fit_intercept=True, normalize=False, precompute=False, copy_X=True, max_iter=1000, tol=tol,
          warm_start=False, positive=False, random_state=None, selection='cyclic'),
    KernelRidge(alpha=1.0, coef0=1, degree=3, gamma=3, kernel='poly'),
    RANSACRegressor(random_state=random_state),
    BayesianRidge(n_iter=300, tol=0.001, alpha_1=1e-06, alpha_2=1e-06, lambda_1=1e-06, lambda_2=1e-06,
                  compute_score=False, fit_intercept=True, normalize=False, copy_X=True, verbose=False),
    ExtraTreesRegressor()
]

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
# print(X_array)
X_label = np.array(df.columns)
X_label = X_label[1:X_label.shape[0]]
X_label = np.array(X_label)

Y_array = np.array(df.loc[:, 'soil_moisture'])
# Y_array = Y_array.reshape(-1,1)
Y_array = np.array(Y_array)

'''
离群值检测
'''
cov = EllipticEnvelope(random_state=1, contamination=0.05)
cov.fit(np.hstack([X_array, Y_array.reshape(-1, 1)]))
index = cov.predict(np.hstack([X_array, Y_array.reshape(-1, 1)]))


def error_wipe(array, index):
    temp_array = []
    for num, line in enumerate(array):
        if index[num] == 1:
            temp_array.append(line)
        elif index[num] == -1:
            # print(line)
            pass
    temp_array = np.array(temp_array)
    return temp_array


X_array = error_wipe(X_array, index)
Y_array = error_wipe(Y_array, index)

'''
标准化
'''
O_X_array = X_array
X_array = StandardScaler().fit_transform(X_array)

'''
样本划分
'''
x_train, x_test, y_train, y_test = train_test_split(X_array, Y_array, train_size=500, random_state=random_state)
line = 5


def continuous_to_multiclass(x_train, x_test, y_train, y_test):
    x_train = (x_train * 1000).astype(int)
    x_test = (x_test * 1000).astype(int)
    y_train = (y_train * 1000).astype(int)
    y_test = (y_test * 1000).astype(int)
    return x_train, x_test, y_train, y_test


# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

# font_cn = FontProperties(fname='simhei.ttf')
# font_en = FontProperties(fname='times.ttf')

plt.figure('multiple regressor', dpi=100, figsize=(21, 9))
plt.suptitle('Multiple Regressor 土壤含水量高光谱数据多模型对比', fontsize=20)
result = []
n = 1
for name, regressor in zip(names, regressors):
    print(name)
    try:
        regressor.fit(x_train, y_train)
    except ValueError:
        p_x_train, p_x_test, p_y_train, p_y_test = continuous_to_multiclass(x_train, x_test, y_train, y_test)
        regressor.fit(p_x_train, p_y_train)
        error = False
        try:
            score = regressor.score(p_x_train, p_y_train)
        except AttributeError:
            score = False
            print(AttributeError)

        try:
            rmse = RMSE(p_y_test, regressor.predict(p_x_test))
        except AttributeError:
            rmse = False
            print(AttributeError)
    else:
        error = True
        try:
            score = regressor.score(x_train, y_train)
        except AttributeError:
            score = False
            print(AttributeError)

        try:
            rmse = RMSE(y_test, regressor.predict(x_test))
        except AttributeError:
            rmse = False
            print(AttributeError)
    result.append([name, error, score, rmse])
    plt.subplot(4, 6, n)
    if error != False and score != False and rmse != False:
        score = r2_score(y_test, regressor.predict(x_test))
        plt.title(name, fontsize=10)
        plt.text(2.5, 39, 'r^2=' + str(np.around(score, decimals=2)) + '\nrmse=' + str(np.around(rmse, decimals=2)),
                 fontsize=10)
        plt.scatter(x_test[:, 50], y_test, label='origin', color='chocolate', marker='.')
        plt.scatter(x_test[:, 50], regressor.predict(x_test), label='predict', color='teal', marker='.')
        # plt.xlabel('abs /A',fontsize=8)
        # plt.ylabel('moisture /%',fontsize=8)
        plt.yticks([25, 30, 35, 40, 45, 50], ['min', '30%', '35%', '40%', '45%', 'max'], fontsize=8, rotation=45)
        plt.xticks([])
        plt.axis([-1, 4, 25, 50])
        plt.legend(fontsize=8)
    elif error == False:
        score = r2_score(p_y_test, regressor.predict(p_x_test))
        plt.title(name, fontsize=10)
        plt.text(2500, 39000,
                 'r^2=' + str(np.around(score, decimals=2)) + '\nrmse=' + str(np.around((rmse / 1000), decimals=2)),
                 fontsize=10)
        plt.scatter(p_x_test[:, 50], p_y_test, label='origin', color='chocolate', marker='.')
        plt.scatter(p_x_test[:, 50], regressor.predict(p_x_test), label='predict', color='teal', marker='.')
        # plt.xlabel('abs /A',fontsize=8)
        # plt.ylabel('moisture /%',fontsize=8)
        plt.yticks([25000, 30000, 35000, 40000, 45000, 50000], ['min', '30%', '35%', '40%', '45%', 'max'], fontsize=8,
                   rotation=45)
        plt.xticks([])
        plt.axis([-1000, 4000, 25000, 50000])
        plt.legend(fontsize=6)
    else:
        plt.title(name, fontsize=8)
        plt.scatter(x_test[:, 50], y_test, color='crimson', label='error', marker='.')
        plt.yticks([25, 30, 35, 40, 45, 50], ['min', '30%', '35%', '40%', '45%', 'max'], fontsize=8, rotation=45)
        plt.axis([-1, 4, 25, 50])
        plt.xticks([])
        plt.legend(fontsize=8)
    n = n + 1
end_time = time.time()
tol_time = str(np.around(end_time - start_time, decimals=2))
for i in result:
    print(i)
print('time =' + tol_time + 's')
plt.savefig('multiple regressor.jpg', dpi=400)
plt.show()
