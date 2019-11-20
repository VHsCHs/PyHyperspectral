import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time
from sklearn.ensemble import IsolationForest
from private_model.pymodel import nine_point_average, RMSE
from sklearn.metrics import r2_score

clf = MLPRegressor(
    hidden_layer_sizes=(500, 30), activation='relu', solver='lbfgs', alpha=0.0001, batch_size='auto',
    learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000, shuffle=True,
    random_state=1, tol=1e-4, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)

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
# print(X_array)
X_label = np.array(df.columns)
X_label = X_label[1:X_label.shape[0]]
X_label = np.array(X_label)
Y_array = np.array(df.loc[:, 'soil_moisture'])
Y_array = np.array(Y_array)

''''
离群值检测
'''
cov = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, max_features=1.0, bootstrap=False,
                      n_jobs=1, random_state=1, verbose=0, behaviour='new')

cov.fit(np.hstack([X_array, Y_array.reshape(-1, 1)]))
index = cov.predict(np.hstack([X_array, Y_array.reshape(-1, 1)]))


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
scaler = StandardScaler()
scaler.fit(X_array)
X_array = scaler.transform(X_array)

'''
样本划分
'''
x_train, x_test, y_train, y_test = train_test_split(X_array, Y_array, train_size=500, random_state=1)

'''
九点加权移动平均
'''
p_x_train = []
p_x_test = []
for i in x_train:
    p_x_train.append(nine_point_average(i))
p_x_train = np.array(p_x_train)
for i in x_test:
    p_x_test.append(nine_point_average(i))
p_x_test = np.array(p_x_test)

clf1 = clf
clf1.fit(x_train, y_train)
clf2 = clf
clf2.fit(p_x_train, y_train)

plt.figure(figsize=(9.6, 5.4), dpi=100)
plt.subplot(2, 1, 1)
plt.title('origin')
plt.scatter(x_test[:, 0], y_test, s=10, c='chocolate', label='origin')
plt.scatter(x_test[:, 0], clf1.predict(x_test), s=3, c='teal', label='predict')
plt.text(2.75, 33,
         'r^2=' + str(round(r2_score(y_test, clf1.predict(x_test)), 3)) +
         '\nrmse=' + str(round(RMSE(y_test, clf1.predict(x_test)), 3)))
plt.legend(fontsize=8)
plt.xticks([]), plt.yticks(fontsize=6)

plt.subplot(2, 1, 2)
plt.title('tranform')
plt.scatter(p_x_test[:, 0], y_test, s=10, c='chocolate', label='origin')
plt.scatter(p_x_test[:, 0], clf2.predict(p_x_test), s=3, c='teal', label='predict')
plt.text(2.75, 33,
         'r^2=' + str(round(r2_score(y_test, clf2.predict(p_x_test)), 3)) +
         '\nrmse=' + str(round(RMSE(y_test, clf2.predict(p_x_test)), 3)))
plt.legend(fontsize=8)
plt.xticks([]), plt.yticks(fontsize=6)

plt.savefig('D:\\Desktop\\nine_point.jpg',dpi=100)
plt.show()