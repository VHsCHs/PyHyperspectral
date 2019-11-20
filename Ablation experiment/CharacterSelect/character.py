import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import os, sys, time, math
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import column_or_1d
from numpy import *
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier

start_time = time.time()

# Classifier = MLPClassifier(
#     solver='lbfgs', activation='relu', alpha=0.0001, hidden_layer_sizes=(500, 100), random_state=1, max_iter=1000000,
#     learning_rate_init=0.0001, learning_rate='constant', tol=1e-4, power_t=0.5, shuffle=True,
#     verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, beta_1=0.9,
#     beta_2=0.999, epsilon=1e-08)

Classifier = RandomForestClassifier(n_estimators=50, criterion='gini', max_depth=None, min_samples_split=27,
                                    min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                    max_features='auto', max_leaf_nodes=None, min_impurity_decrease=0.0,
                                    min_impurity_split=None, bootstrap=True, oob_score=False, n_jobs=1, random_state=1,
                                    verbose=0, warm_start=False, class_weight=None)

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

# 读取数据
file = 'D:\\Desktop\\datasave.csv'
with open(file, 'r') as f:
    data = pd.read_csv(f, index_col=[0], header=[0])

# data = data.query('moisture == 0')
ox = data.loc[:, '404.7':'1010.8']
ox = np.array(ox)
oy = data.loc[:, 'moisture':'moisture']
oy = np.array(oy)
oy = oy.ravel()
oy = column_or_1d(oy, warn=True)

# plt支持中文
plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号

plt.figure(figsize=(9.6, 5.4), dpi=100)
plt.suptitle('特征筛选')
num = 1

'''
1. 移除低方差的特征 (Removing features with low variance)
'''
print(num)
from sklearn.feature_selection import VarianceThreshold

plt.subplot(2, 3, num)
num = num + 1

Var = VarianceThreshold(threshold=0.0032)
Var.fit(ox)
x = Var.transform(ox)
print(all_np(Var.get_support()))

x_train, x_test, y_train, y_test = train_test_split(x, oy, test_size=0.25, random_state=1)
cls1 = Classifier
cls1.fit(x_train, y_train)

# print(str(round(accuracy_score(y_test, cls1.predict(x_test)), 3)))

plt.scatter(x_train[:, 0], y_train, label='origin', c='b', s=10)
plt.scatter(x_train[:, 0], cls1.predict(x_train), label='predict', c='r', s=3)
plt.xticks([]), plt.yticks(fontsize=6)
plt.title('移除低方差的特征 score=' + str(round(accuracy_score(y_test, cls1.predict(x_test)), 3)))
plt.text(0.58,22,'选择特征数:'+ str(all_np(Var.get_support())[True]))
plt.legend(fontsize=6)

'''
2. 卡方检验 (chi2)
For classification: chi2, f_classif, mutual_info_classif
For regression: f_regression, mutual_info_regression
'''
print(num)
from sklearn.feature_selection import SelectKBest, SelectPercentile
from sklearn.feature_selection import chi2, f_regression

plt.subplot(2, 3, num)
num = num + 1
# x = SelectPercentile(f_regression, k=2).fit_transform(ox, oy)

kBest = SelectKBest(chi2, k=5)
kBest.fit(ox, oy)
x = kBest.transform(ox)
print(all_np(kBest.get_support()))

x_train, x_test, y_train, y_test = train_test_split(x, oy, test_size=0.25, random_state=1)
cls2 = Classifier
cls2.fit(x_train, y_train)
print(str(round(accuracy_score(y_test, cls2.predict(x_test)), 3)))
plt.scatter(x_train[:, 0], y_train, label='origin', c='b', s=10)
plt.scatter(x_train[:, 0], cls2.predict(x_train), label='predict', c='r', s=3)
plt.xticks([]), plt.yticks(fontsize=6)
plt.title('卡方检验 score=' + str(round(accuracy_score(y_test, cls2.predict(x_test)), 3)))
plt.text(0.28,22,'选择特征数:'+ str(all_np(kBest.get_support())[True]))
plt.legend(fontsize=6)

'''
3.  相关系数法 (Pearson R)
'''
print(num)
plt.subplot(2, 3, num)
num = num + 1

from scipy.stats import pearsonr

rBest = SelectKBest(lambda X, Y: tuple(map(tuple, array(list(map(lambda x: pearsonr(x, Y), X.T))).T)), k=60)
rBest.fit(ox, oy)
x = rBest.transform(ox)
print(all_np(rBest.get_support()))
x_train, x_test, y_train, y_test = train_test_split(x, oy, test_size=0.25, random_state=1)
cls3 = Classifier
cls3.fit(x_train, y_train)
plt.scatter(x_train[:, 0], y_train, label='origin', c='b', s=10)
plt.scatter(x_train[:, 0], cls3.predict(x_train), label='predict', c='r', s=3)
plt.xticks([]), plt.yticks(fontsize=6)
plt.title('相关系数法 score=' + str(round(accuracy_score(y_test, cls3.predict(x_test)), 3)))
plt.text(0.28,22,'选择特征数:'+ str(all_np(rBest.get_support())[True]))
plt.legend(fontsize=6)

'''
4. 递归特征消除 (Recursive Feature Elimination)
'''
print(num)
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier

plt.subplot(2, 3, num)
num = num + 1

rfe = RFE(estimator=GradientBoostingClassifier(learning_rate=0.1, n_estimators=100, subsample=1.0,
                                               criterion='friedman_mse',
                                               min_samples_split=2, min_samples_leaf=1, min_weight_fraction_leaf=0.0,
                                               max_depth=3,
                                               min_impurity_decrease=0.0, min_impurity_split=None, init=None,
                                               random_state=1,
                                               max_features=None, verbose=0, max_leaf_nodes=None,
                                               warm_start=False,
                                               presort='auto', validation_fraction=0.1, n_iter_no_change=None,
                                               tol=1e-04), n_features_to_select=10)
rfe.fit(ox, oy)
x = rfe.transform(ox)
print(all_np(rfe.get_support()))
x_train, x_test, y_train, y_test = train_test_split(x, oy, test_size=0.25, random_state=1)
cls4 = Classifier
cls4.fit(x_train, y_train)
plt.scatter(x_train[:, 0], y_train, label='origin', c='b', s=10)
plt.scatter(x_train[:, 0], cls4.predict(x_train), label='predict', c='r', s=3)
plt.xticks([]), plt.yticks(fontsize=6)
plt.title('递归特征消除 score=' + str(round(accuracy_score(y_test, cls4.predict(x_test)), 3)))
plt.text(0.28,22,'选择特征数:'+ str(all_np(rfe.get_support())[True]))
plt.legend(fontsize=6)

'''
5. 基于惩罚项的特征选择法 (Feature selection using SelectFromModel)
'''
print(num)
from sklearn.svm import LinearSVC
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import SGDClassifier

plt.subplot(2, 3, num)
num = num + 1

# lsvc = LinearSVC(C=0.001, penalty="l1", dual=False)
lsvc = SGDClassifier(loss='squared_loss', penalty='l1', alpha=0.0001, l1_ratio=0.15, fit_intercept=True, max_iter=1000,
                     tol=1e-4, shuffle=True, verbose=0, epsilon=0.1, random_state=1,
                     learning_rate='invscaling',
                     eta0=0.01, power_t=0.25, early_stopping=False, validation_fraction=0.1, n_iter_no_change=5,
                     warm_start=False, average=False)
lsvc.fit(ox, oy)
model = SelectFromModel(lsvc, prefit=True)
x = model.transform(ox)
print(all_np(model.get_support()))
x_train, x_test, y_train, y_test = train_test_split(x, oy, test_size=0.25, random_state=1)
cls5 = Classifier
cls5.fit(x_train, y_train)
plt.scatter(x_train[:, 0], y_train, label='origin', c='b', s=10)
plt.scatter(x_train[:, 0], cls5.predict(x_train), label='predict', c='r', s=3)
plt.xticks([]), plt.yticks(fontsize=6)
plt.title('基于惩罚项的特征选择法 score=' + str(round(accuracy_score(y_test, cls5.predict(x_test)), 3)))
plt.text(0.265,22,'选择特征数:'+ str(all_np(model.get_support())[True]))
plt.legend(fontsize=6)

'''
6. 基于树的特征选择 (Tree-based feature selection)
'''
print(num)
plt.subplot(2, 3, num)
num = num + 1

from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel

clf = ExtraTreesClassifier(n_estimators=1000)
clf = clf.fit(ox, oy)
model = SelectFromModel(clf, prefit=True)
x = model.transform(ox)
print(all_np(model.get_support()))
x_train, x_test, y_train, y_test = train_test_split(x, oy, test_size=0.25, random_state=1)
cls6 = Classifier
cls6.fit(x_train, y_train)
plt.scatter(x_train[:, 0], y_train, label='origin', c='b', s=10)
plt.scatter(x_train[:, 0], cls6.predict(x_train), label='predict', c='r', s=3)
plt.xticks([]), plt.yticks(fontsize=6)
plt.title('基于树的特征选择 score=' + str(round(accuracy_score(y_test, cls6.predict(x_test)), 3)))
plt.text(0.28,22,'选择特征数:'+ str(all_np(model.get_support())[True]))
plt.legend(fontsize=6)

plt.savefig('D:\\Desktop\\character.jpg', dpi=200)
end_time = time.time()
tol_time = str(round(end_time - start_time, 3))
print('total time=' + tol_time + 's')

plt.show()
# '''
# 7. 将特征选择过程融入pipeline (Feature selection as part of a pipeline)
# '''
#
# clf = Pipeline([
#     ('feature_selection', SelectFromModel(LinearSVC(penalty="l1"))),
#     ('classification', RandomForestClassifier())
# ])
# clf.fit(X, y)
