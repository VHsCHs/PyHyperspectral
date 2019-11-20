#!/usr/bin/env python
# -*- coding: utf-8 -*-

import time
import pandas as pd
from numpy import *
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.ensemble import IsolationForest, RandomForestRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.pipeline import Pipeline
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.feature_selection import SelectKBest
from private_model.pymodel import RMSE,RRMSE ,all_np, error_wipe, NinePointAverage, FirstDerivatives, MultipleScatterCorrection, \
    SavitzkyGolayFilter, HarrDwt, str_round

start_time = time.time()

# 支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

file = 'D:\\Desktop\\datasave_without_smooth.csv'
with open(file, 'r') as f:
    data = pd.read_csv(f, index_col=[0], header=[0])
    x_array = data.loc[:, '404.7': '1010.8']
    y_array = data.loc[:, 'moisture']
    x_label = x_array.columns
    x_array, y_array, x_label = np.array(x_array), np.ravel(np.array(y_array)), np.array(x_label)
random_seed = 10

SVM_1 = Pipeline([
    ('S-G Filter', SavitzkyGolayFilter()),
    ('SVM', SVR(kernel='poly', degree=3, gamma='scale', coef0=0.0, tol=1e-04, C=0.8, epsilon=0.1, shrinking=True,
                cache_size=200, verbose=False, max_iter=-1))
])
SVM_2 = Pipeline([
    ('SVM', SVR(kernel='poly', degree=3, gamma='scale', coef0=0.0, tol=1e-04, C=0.8, epsilon=0.1, shrinking=True,
                cache_size=200, verbose=False, max_iter=-1))
])
SVM_3 = Pipeline([
    ('FirstDerivatives', FirstDerivatives(x_label)),
    ('Feature Selection_1',
     # SelectKBest(chi2, k=125)
     SelectKBest(lambda X, Y: tuple(map(tuple, array(list(map(lambda x: pearsonr(x, Y), X.T))).T)), k=75)

     # RFE(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
     #                                          criterion='friedman_mse',
     #                                          min_samples_split=2, min_samples_leaf=1,
     #                                          min_weight_fraction_leaf=0.0,
     #                                          max_depth=3,
     #                                          min_impurity_decrease=0.0, min_impurity_split=None,
     #                                          init=None,
     #                                          random_state=random_seed,
     #                                          max_features=None, verbose=0, max_leaf_nodes=None,
     #                                          warm_start=False,
     #                                          presort='auto', validation_fraction=0.1,
     #                                          n_iter_no_change=None,
     #                                          tol=1e-04), n_features_to_select=75)
     ),
    ('Haar Dwt', HarrDwt(iterations=3)),
    ('SVM', SVR(kernel='poly', degree=3, gamma='scale', coef0=0.0, tol=1e-04, C=1.0, epsilon=0.1, shrinking=True,
                cache_size=200, verbose=False, max_iter=-1))
])
SVM_4 = Pipeline([
    ('FirstDerivatives', FirstDerivatives(x_label)),
    ('Feature Selection_2',
     # SelectKBest(chi2, k=25)
     SelectKBest(lambda X, Y: tuple(map(tuple, array(list(map(lambda x: pearsonr(x, Y), X.T))).T)), k=50)
     # RFE(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
     #                                          criterion='friedman_mse',
     #                                          min_samples_split=2, min_samples_leaf=1,
     #                                          min_weight_fraction_leaf=0.0,
     #                                          max_depth=3,
     #                                          min_impurity_decrease=0.0, min_impurity_split=None,
     #                                          init=None,
     #                                          random_state=random_seed,
     #                                          max_features=None, verbose=0, max_leaf_nodes=None,
     #                                          warm_start=False,
     #                                          presort='auto', validation_fraction=0.1,
     #                                          n_iter_no_change=None,
     #                                          tol=1e-04), n_features_to_select=50)
     ),
    ('Haar Dwt', HarrDwt(iterations=3)),
    ('SVM', SVR(kernel='poly', degree=3, gamma='scale', coef0=0.0, tol=1e-04, C=1.0, epsilon=0.1, shrinking=True,
                cache_size=200, verbose=False, max_iter=-1))
])
MLP_1 = Pipeline([
    ('Haar Dwt', HarrDwt(iterations=3)),
    ('S-G filter', SavitzkyGolayFilter()),
    ('MLP', MLPRegressor(hidden_layer_sizes=(400, 40), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.00001, power_t=0.5, max_iter=100000,
                         shuffle=True,
                         random_state=random_seed, tol=1e-6, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
])
MLP_2 = Pipeline([
    ('Haar Dwt', HarrDwt(iterations=4)),
    ('PCA', PCA(n_components='mle')),
    ('MLP', MLPRegressor(hidden_layer_sizes=(500, 50), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=20000,
                         shuffle=True,
                         random_state=random_seed, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
])
MLP_3 = Pipeline([
    ('Haar Dwt', HarrDwt(iterations=3)),
    ('MLP', MLPRegressor(hidden_layer_sizes=(400,), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=5000,
                         shuffle=True,
                         random_state=random_seed, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
])
MLP_4 = Pipeline([
    ('Haar Dwt', HarrDwt(iterations=3)),
    ('MLP', MLPRegressor(hidden_layer_sizes=(500,), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=5000,
                         shuffle=True,
                         random_state=random_seed, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
])
MLP_5 = Pipeline([
    ('FirstDerivatives', FirstDerivatives(x_label)),
    ('Haar Dwt', HarrDwt(iterations=2)),
    ('Feature Selection_3',
     # SelectKBest(chi2, k=75)
     SelectKBest(lambda X, Y: tuple(map(tuple, array(list(map(lambda x: pearsonr(x, Y), X.T))).T)), k=25)
     # RFE(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
     #                                          criterion='friedman_mse',
     #                                          min_samples_split=2, min_samples_leaf=1,
     #                                          min_weight_fraction_leaf=0.0,
     #                                          max_depth=3,
     #                                          min_impurity_decrease=0.0, min_impurity_split=None,
     #                                          init=None,
     #                                          random_state=random_seed,
     #                                          max_features=None, verbose=0, max_leaf_nodes=None,
     #                                          warm_start=False,
     #                                          presort='auto', validation_fraction=0.1,
     #                                          n_iter_no_change=None,
     #                                          tol=1e-04), n_features_to_select=25)
     ),
    ('MLP', MLPRegressor(hidden_layer_sizes=(500, 50), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=5000,
                         shuffle=True,
                         random_state=random_seed, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
])
MLP_6 = Pipeline([
    ('FirstDerivatives', FirstDerivatives(x_label)),
    ('Haar Dwt', HarrDwt(iterations=4)),
    ('MLP', MLPRegressor(hidden_layer_sizes=(500, 50), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=10000,
                         shuffle=True,
                         random_state=random_seed, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
])
MLP_7 = Pipeline([
    ('Haar Dwt', HarrDwt(iterations=4)),
    ('MLP', MLPRegressor(hidden_layer_sizes=(500, 50), activation='relu', solver='lbfgs', alpha=0.0001,
                         batch_size='auto',
                         learning_rate='constant', learning_rate_init=0.0001, power_t=0.5, max_iter=5000,
                         shuffle=True,
                         random_state=random_seed, tol=1e-4, verbose=False, warm_start=False, momentum=0.9,
                         nesterovs_momentum=True,
                         early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08))
])
TREE_1 = Pipeline([
    ('Random Forest',
     RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=10,
                           min_samples_split=2, min_samples_leaf=1,
                           min_weight_fraction_leaf=0.0, max_features=1,
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, bootstrap=True, oob_score=True,
                           n_jobs=None, random_state=random_seed, verbose=0, warm_start=False)
     # GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
     #                          normalize_y=True, copy_X_train=True, random_state=random_seed)
     )
])
TREE_2 = Pipeline([
    ('MSC', MultipleScatterCorrection(iterations=2)),
    ('Random Forest',
     RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=10,
                           min_samples_split=2, min_samples_leaf=1,
                           min_weight_fraction_leaf=0.0, max_features=1,
                           max_leaf_nodes=None, min_impurity_decrease=0.0,
                           min_impurity_split=None, bootstrap=True, oob_score=True,
                           n_jobs=None, random_state=random_seed, verbose=0, warm_start=False)
     # GaussianProcessRegressor(kernel=None, alpha=1e-10, optimizer='fmin_l_bfgs_b', n_restarts_optimizer=0,
     #                          normalize_y=True, copy_X_train=True, random_state=random_seed)
     )
])
MVR_1 = Pipeline([
    ('Feature Selection_4',
     # SelectKBest(chi2, k=50)
     SelectKBest(lambda X, Y: tuple(map(tuple, array(list(map(lambda x: pearsonr(x, Y), X.T))).T)), k=50)
     # RFE(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
     #                                          criterion='friedman_mse',
     #                                          min_samples_split=2, min_samples_leaf=1,
     #                                          min_weight_fraction_leaf=0.0,
     #                                          max_depth=3,
     #                                          min_impurity_decrease=0.0, min_impurity_split=None,
     #                                          init=None,
     #                                          random_state=random_seed,
     #                                          max_features=None, verbose=0, max_leaf_nodes=None,
     #                                          warm_start=False,
     #                                          presort='auto', validation_fraction=0.1,
     #                                          n_iter_no_change=None,
     #                                          tol=1e-04), n_features_to_select=50)
     ),
    ('PLSRegression', PLSRegression(n_components=10)
     )
])
MVR_2 = Pipeline([
    ('Haar Dwt', HarrDwt(iterations=3)),
    ('PLSRegression', PLSRegression(n_components=10))
])

models = [SVM_1, SVM_2, SVM_3, SVM_4, MLP_1, MLP_2, MLP_3, MLP_4, MLP_5, MLP_6, MLP_7, TREE_1, TREE_1, MVR_1, MVR_2]
model_name = ['SVM_1', 'SVM_2', 'SVM_3', 'SVM_4', 'MLP_1', 'MLP_2', 'MLP_3', 'MLP_4', 'MLP_5', 'MLP_6', 'MLP_7',
              'TREE_1', 'TREE_2', 'MVR_1', 'MVR_2']
# model_weight = [0.127, 0.038, 0.063, 0.063, 0.127, 0.127, 0.013, 0.089, 0.152, 0.038, 0.076, 0.019, 0.013, 0.019, 0.013]
model_weight = [0.127, 0.089, 0.063, 0.063, 0.039, 0.038, 0.089, 0.076, 0.049, 0.038, 0.049, 0.013, 0.013, 0.127, 0.127]
print(np.sum(model_weight))
OutlierRemover = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, max_features=1.0,
                                 bootstrap=False,
                                 n_jobs=1, random_state=1, verbose=0, behaviour='new')
OutlierRemover.fit(np.hstack([x_array, y_array.reshape(-1, 1)]))
index = OutlierRemover.predict(np.hstack([x_array, y_array.reshape(-1, 1)]))
x_array = error_wipe(x_array, index)
y_array = error_wipe(y_array, index)
print(all_np(index))

x_train, x_test, y_train, y_test = train_test_split(x_array, y_array, train_size=0.75, random_state=random_seed)
print(np.shape(x_train))
print(np.shape(y_train))

predict_, rmse_, r2_score_ = [], [], []
for num, model in enumerate(models):
    model_start = time.time()
    model.fit(x_train, y_train)
    predict_.append(np.ravel(model.predict(x_test)))
    r2_score_.append(r2_score(y_test, model.predict(x_test)))
    rmse_.append(RMSE(y_test, model.predict(x_test)))
    model_end = time.time()
    model_tol = str_round(model_end - model_start)
    print('\nmodel name:' + model_name[num])
    print('R^2= ' + str_round(r2_score(y_test, model.predict(x_test)), 3))
    print('RMSE= ' + str_round(math.sqrt(mean_squared_error(y_test, model.predict(x_test))), 3))
    print('RRMSE= ' + str_round(RRMSE(y_test, model.predict(x_test)), 3))
    print('model running time=' + model_tol + 's')

predict_ = np.array(predict_)
print(predict_[:, 6])
y_predict = np.average(predict_, weights=model_weight, axis=0)
print(y_predict)
print(y_test)
print(str_round(r2_score(y_test, y_predict), 3))
print(str_round(math.sqrt(mean_squared_error(y_test, y_predict)), 3))

print('\nNine Point Average\n')
nine = NinePointAverage()
nine.fit(x_train)
x_train = nine.transform(x_train)
x_test = nine.transform(x_test)

predict_, rmse_, r2_score_ = [], [], []
for num, model in enumerate(models):
    model_start = time.time()
    model.fit(x_train, y_train)
    predict_.append(np.ravel(model.predict(x_test)))
    r2_score_.append(r2_score(y_test, model.predict(x_test)))
    rmse_.append(RMSE(y_test, model.predict(x_test)))
    model_end = time.time()
    model_tol = str_round(model_end - model_start)
    print('\nmodel name:' + model_name[num])
    print('R^2= ' + str_round(r2_score(y_test, model.predict(x_test)), 3))
    print('RMSE= ' + str_round(math.sqrt(mean_squared_error(y_test, model.predict(x_test))), 3))
    print('RRMSE= ' + str_round(RRMSE(y_test, model.predict(x_test)), 3))
    print('model running time=' + model_tol + 's')

predict_ = np.array(predict_)
print(predict_[:, 6])
y_predict = np.average(predict_, weights=model_weight, axis=0)
print(y_predict)
print(y_test)
print(str_round(r2_score(y_test, y_predict), 3))
print(str_round(math.sqrt(mean_squared_error(y_test, y_predict)), 3))

end_time = time.time()
tol_time = str_round(end_time - start_time, 3)
print('total time=' + tol_time + 's')
