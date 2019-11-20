import csv
from models import csv_read
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cross_decomposition import PLSRegression, PLSCanonical, CCA
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor

csv_file = "D:\\Desktop\\PyProject\\save_without_WT.csv"
csv_data = csv_read.csv_reader(csv_file)
var_name = csv_read.csv_varname(csv_file)
csv_shape = np.shape(csv_data)
csv_line = csv_shape[0]
csv_var = csv_shape[1]
point_index = var_name.index('point_1')
sub_index = var_name.index('TOC')
WT_index = var_name.index('WT')
# print(csv_read.csv_samplename(csv_file))
# print(csv_data)
average_data = csv_read.csv_average(csv_data, csv_file)
print(average_data)

PLS_Regression(average_data, point_index, sub_index, var_name)
Ordinary_Least_Square(average_data, point_index, sub_index, var_name)
PLS_CCA(average_data, point_index, sub_index, var_name)
PLS_Canonical(average_data, point_index, sub_index, var_name, components=9)
Linear_Regression(average_data, point_index, sub_index, var_name)
RMSE(model.predict, average_data)
MLP_Regressor(average_data, point_index, sub_index, var_name)
Abnormal_Data(average_data, point_index, sub_index, var_name)


def Linear_Regression(csv_data, point_index, sub_index, var_name, train=None, components=None):
    R_array = []
    for j in range(7):
        temp_array = {}
        temp_array["var_name"] = var_name[sub_index + j]
        # print(var_name[sub_index + j])
        for i in range(8):
            x = np.array(csv_data[:, point_index - 1 + i])
            # y = np.array(csv_data[:, sub_index -1 + j] / csv_data[:, WT_index - 1])
            y = np.array(csv_data[:, sub_index - 1 + j])
            # y = math.log(y)
            x = x.reshape(-1, 1)
            y = y.reshape(-1, 1)
            y = np.log(y)
            # plt.scatter(x,y)
            # plt.show()
            lrModel = LinearRegression()
            lrModel.fit(x, y)
            temp_array[var_name[point_index + i]] = lrModel.score(x, y)
            # print(lrModel.score(x,y))
        R_array.append(temp_array)
        print(R_array)
        sub_title = []
        sub_title.append('var_name')
        for i in range(9):
            sub_title.append(var_name[point_index + i])
        print(sub_title)
        csv_wirte_file = "D:\\Desktop\\PyProject\\R_save_without_WT_with_log.csv"
        with open(csv_wirte_file, "w", newline="") as data_write:
            data_writer = csv.DictWriter(data_write, fieldnames=sub_title)
            data_writer.writeheader()
            data_writer.writerows(R_array)


def Ordinary_Least_Square(csv_data, point_index, sub_index, var_name, train=None, components=None):
    '''
    plt initinitalize & definition
    '''
    plt.figure()

    X_array = []
    temp_array = []
    for j in csv_data:
        temp_array = j[point_index - 1:point_index + 8]
        X_array.append(temp_array)
    X_array = np.array(X_array)
    if components == None:
        pass
    else:
        pca = PCA(n_components=components)
        X_array = pca.fit_transform(X_array)
    if train == True:
        X_array, X_test, Y_array, Y_test = train_test_split(X_array, Y_array, test_size=0.25, random_state=42)
    for i in range(7):
        Y_array = np.array(csv_data[:, sub_index - 1 + i])
        lrModel = LinearRegression()
        lrModel.fit(X_array, Y_array)
        coefs = lrModel.coef_
        coefs = np.around(coefs, decimals=2)
        coefs = coefs.astype(str)
        print(var_name[sub_index + i])
        print("y =", end="")
        for j in range(9):
            print(coefs[j], end="")
            print("*x", end="")
            print(j + 1, end="")
            print(" + ", end="")
        print(np.around(lrModel.intercept_, decimals=2))
        print("R^2=", np.around(lrModel.score(X_array, Y_array), decimals=2))
        print('RMSE=', RMSE(Y_array, lrModel.predict(X_array)))
        Y_predict = lrModel.predict(X_array)
        plt.subplot(3, 3, i + 1)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        score = np.around(lrModel.score(X_array, Y_array), decimals=2)
        plt.text(0.6, 10, "R^2 =" + str(score), horizontalalignment='center', verticalalignment='center')
        plt.title(var_name[sub_index + i])
        plt.scatter(X_array[:, 0], Y_array)
        plt.scatter(X_array[:, 0], Y_predict)
    plt.suptitle('Ordinary Least Square')
    plt.show()


def PLS_Regression(csv_data, point_index, sub_index, var_name, train=None, components=None):
    '''
    plt initinitalize & definition
    '''
    plt.figure()

    # plt.subplot(3, 3, 1)
    # plt.plot([0, 1], [0, 1])
    # plt.subplot(3, 3, 2)
    # plt.plot([0, 1], [0, 2])
    # plt.subplot(3, 3, 3)
    # plt.plot([0, 3], [0, 4])
    # plt.subplot(3, 3, 4)
    # plt.plot([0, 1], [0, 2])
    # plt.subplot(3, 3, 5)
    # plt.plot([0, 1], [0, 1])
    # plt.subplot(3, 3, 6)
    # plt.plot([0, 1], [0, 1])
    # plt.subplot(3, 1, 3)
    # plt.plot([0, 1], [0, 3])
    # plt.show()
    for i in range(7):
        X_array = []
        temp_array = []

        for j in csv_data:
            temp_array = j[point_index - 1:point_index + 8]
            X_array.append(temp_array)
        X_array = np.array(X_array)
        Y_array = np.array(csv_data[:, sub_index - 1 + i])
        if train == True:
            X_array, X_test, Y_array, Y_test = train_test_split(X_array, Y_array, test_size=0.15, random_state=42)
        if components == None:
            components = np.shape(X_array)[1]
        plsrModel = PLSRegression(n_components=components)
        plsrModel.fit(X_array, Y_array)
        coefs = plsrModel.coef_
        coefs = np.around(coefs, decimals=2)
        coefs = coefs.astype(str)
        # print(var_name[sub_index + i])
        # print("y =",end="")
        # for i in range(9):
        #     print(coefs[i][0],end="")
        #     print("*x",end="")
        #     print(i+1,end="")
        #     if i != 8:
        #         print(" + ",end="")
        # print('')
        # print("R^2 =",np.around(plsrModel.score(X_array, Y_array),decimals=2))
        Y_predict = plsrModel.predict(X_array)
        plt.subplot(3, 3, i + 1)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        score = np.around(plsrModel.score(X_array, Y_array), decimals=2)
        plt.text(0.6, 10, "R^2 =" + str(score), horizontalalignment='center', verticalalignment='center')
        plt.title(var_name[sub_index + i])
        plt.scatter(X_array[:, 0], Y_array)
        plt.scatter(X_array[:, 0], Y_predict)
        print(RMSE(Y_array, plsrModel.predict(X_array)))
    plt.suptitle('PLS-Regression')
    plt.show()


def PLS_CCA(csv_data, point_index, sub_index, var_name, train=None, components=None):
    X_array = []
    temp_array = []
    for j in csv_data:
        temp_array = j[point_index - 1:point_index + 8]
        X_array.append(temp_array)
    X_array = np.array(X_array)
    if components == None:
        components = np.shape(X_array)[1]
    for i in range(7):
        Y_array = np.array(csv_data[:, sub_index - 1 + i])
        ccaModel = CCA(n_components=1)
        ccaModel.fit(X_array, Y_array)
        print(var_name[sub_index + i])
        print("R^2 =", np.around(ccaModel.score(X_array, Y_array), decimals=2))
        # X_train_r, Y_train_r = cca.transform(X_train, Y_train)
        # X_test_r, Y_test_r = cca.transform(X_test, Y_test)


def PLS_Canonical(csv_data, point_index, sub_index, var_name, train=None, components=None):
    X_array = []
    temp_array = []
    for j in csv_data:
        temp_array = j[point_index - 1:point_index + 8]
        X_array.append(temp_array)
    X_array = np.array(X_array)
    if components == None:
        components = np.shape(X_array)[1]
    for i in range(7):
        Y_array = np.array(csv_data[:, sub_index - 1 + i])
        plsca = PLSCanonical(n_components=1)
        plsca.fit(X_array, Y_array)
        print(var_name[sub_index + i])
        print("R^2 =", np.around(plsca.score(X_array, Y_array), decimals=2))
        # X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
        # X_test_r, Y_test_r = plsca.transform(X_test, Y_test)


def MLP_Regressor(csv_data, point_index, sub_index, var_name, train=None, components=None):
    '''
    plt initinitalize & definition
    '''
    plt.figure()

    X_array = []
    for j in csv_data:
        temp_array = j[point_index - 1:point_index + 8]
        X_array.append(temp_array)
    X_array = np.array(X_array)
    scaler = StandardScaler()
    scaler.fit(X_array)
    X_array = scaler.transform(X_array)
    if components == None:
        components = np.shape(X_array)[1]
    for i in range(7):
        Y_array = np.array(csv_data[:, sub_index - 1 + i])
        Mahalanobis_Distace(X_array, Y_array)
        # X_train_r, Y_train_r = plsca.transform(X_train, Y_train)
        # X_test_r, Y_test_r = plsca.transform(X_test, Y_test)
        # clf = MLPRegressor(solver='lbfgs', alpha=1e-6, hidden_layer_sizes=(9, 3), random_state=1)
        clf = MLPRegressor(
            hidden_layer_sizes=(5, 3), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
            learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=15000, shuffle=True,
            random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
            early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
        clf.fit(X_array, Y_array)
        Y_predict = clf.predict(X_array)
        plt.subplot(3, 3, i + 1)
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.title(var_name[sub_index + i])
        plt.scatter(X_array[:, 0], Y_array)
        plt.scatter(X_array[:, 0], Y_predict)
        print(var_name[sub_index + i])
        print('R^2=', clf.score(X_array, Y_array))
        print('RMSE=', RMSE(Y_array, Y_predict))
    plt.suptitle('MLP Regressor')
    plt.show()
    # cengindex = 0
    # for wi in clf.coefs_:
    #     cengindex += 1  # 表示底第几层神经网络。
    #     print('第%d层网络层:' % cengindex)
    #     print('权重矩阵维度:', wi.shape)
    #     print('系数矩阵：\n', wi)
