import numpy as np
import math


def SD(origin_data, predict_data):
    '''
    标准偏差
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.ravel(origin_data)
    predict_data = np.ravel(predict_data)
    return math.sqrt(np.sum(np.square(predict_data - origin_data.mean())) / (origin_data.shape[0] - 1))


def RSD(origin_data, predict_data):
    '''
    相对标准偏差
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.ravel(origin_data)
    predict_data = np.ravel(predict_data)
    return math.sqrt(
        np.sum(np.square(predict_data - origin_data.mean())) / (origin_data.shape[0] - 1)) / origin_data.mean()


def MAE(origin_data, predict_data):
    '''
    平均绝对误差
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.ravel(origin_data)
    predict_data = np.ravel(predict_data)
    return np.mean(np.absolute(predict_data - origin_data))


def RRMSE(origin_data, predict_data):
    '''
    相对均方根误差
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.ravel(origin_data)
    predict_data = np.ravel(predict_data)
    return math.sqrt(np.mean(np.square(origin_data - predict_data))) / np.mean(origin_data)


def RMSE(origin_data, predict_data):
    '''
    均方根误差
    √[∑di^2/n]=Re
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.ravel(origin_data)
    predict_data = np.ravel(predict_data)
    return math.sqrt(np.mean(np.square(origin_data - predict_data)))


def Bias(origin_data, predict_data):
    '''
    Bias乖离率
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.ravel(origin_data)
    predict_data = np.ravel(predict_data)
    return np.mean(predict_data - origin_data.mean())


def R2_coef(origin_data, predict_data):
    '''
    决定系数R2
    :param origin_data:
    :param predict_data:
    :return:
    '''
    origin_data = np.ravel(origin_data)
    predict_data = np.ravel(predict_data)
    return 1 - (np.sum(np.square(predict_data - origin_data)) / np.sum(predict_data - origin_data.mean()))


def R_coef(x, y):
    '''
    相关系数R
    :param x:
    :param y:
    :return:
    '''
    x = np.ravel(x)
    y = np.ravel(y)
    return np.sum(np.multiply(x - x.mean(), y - y.mean())) / \
           math.sqrt(np.sum(np.square(x - x.mean()))) / math.sqrt(np.sum(np.square(y - y.mean())))


def str_round(text, decimal=3):
    return str(round(text, decimal))


def nine_point_average(origin):
    origin = np.array(origin)
    weights = [0.04, 0.08, 0.12, 0.16, 0.20, 0.16, 0.12, 0.08, 0.04]
    weights = np.array(weights)
    revised = []
    for num, point in enumerate(origin):
        if num - 4 < 0:
            revised.append(point)
        elif num + 5 > np.shape(origin)[0]:
            revised.append(point)
        else:
            point_array = np.array(origin[num - 4:num + 5])
            point_array = np.multiply(point_array, weights)
            revised.append(np.sum(point_array, axis=0))
    revised = np.array(revised)
    return revised


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
    valid = np.array(valid)
    return valid


def floatrange(start, stop, step):
    steps = math.floor((stop - start) / step)
    temp = []
    for i in range(steps):
        temp.append(start + step * i)
    return temp


from sklearn.base import BaseEstimator, TransformerMixin


class NinePointAverage(BaseEstimator, TransformerMixin):
    def __init__(self):
        pass

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None, weights='default'):
        X = np.array(X)
        if isinstance(weights, (str, list)):
            if weights == 'default':
                weights = [0.04, 0.08, 0.12, 0.16, 0.20, 0.16, 0.12, 0.08, 0.04]
            elif isinstance(weights, list):
                weights = weights
        weights = np.array(weights)
        average = []
        for line in X:
            linearray = []
            for num, point in enumerate(line):
                if num - 4 > 0 and num + 4 < np.shape(line)[0]:
                    point_array = np.array(line[num - 4:num + 5])
                    point_array = np.average(point_array, weights=weights)
                    linearray.append(point_array)
                else:
                    linearray.append(point)
            average.append(linearray)
        average = np.array(average)
        return average


class FirstDerivatives(BaseEstimator, TransformerMixin):
    def __init__(self, x_label):
        self._x_label = x_label

    def fit(self, X, y=None):
        return self

    def transform(self, x, y=None):
        Y = np.array(x)
        X = np.array(self._x_label)
        # if X.shape[1] == 1:
        #     X = X.ravel()
        revised = []
        for num, point in enumerate(Y):
            if num + 1 >= np.shape(Y)[1]:
                revised.append((point[num + 1] - point[num]) / (X[num + 1] - X[num]))
            else:
                revised.append(point)
        revised = np.array(revised)
        return revised


class SpectralDataTransform(BaseEstimator, TransformerMixin):
    def __init__(self, method, x_label=None):
        self.method = method
        self._x_label = x_label

    def fit(self, X, y=None):
        return self

    def transform(self, x, y=None):
        X = np.array(x)
        X_label = np.array(self._x_label)
        if self.method == 'Inverse' or self.method == 1:
            return np.reciprocal(X)
        elif self.method == 'Log' or self.method == 2:
            return np.log10(X)
        elif self.method == 'Log-Inverse' or self.method == 3:
            return np.reciprocal(np.log10(X))
        elif self.method == 'Sqrt' or self.method == 4:
            return np.sqrt(X)
        elif self.method == 'FirstDerivatives' or self.method == 5:
            revised = []
            for num, point in enumerate(X):
                if num + 1 >= np.shape(X)[1]:
                    revised.append((point[num + 1] - point[num]) / (X_label[num + 1] - X_label[num]))
                else:
                    revised.append(point)
            return np.array(revised)
        elif self.method == 'Inverse-FirstDerivatives' or self.method == 6:
            X = np.reciprocal(X)
            revised = []
            for num, point in enumerate(X):
                if num + 1 >= np.shape(X)[1]:
                    revised.append((point[num + 1] - point[num]) / (X_label[num + 1] - X_label[num]))
                else:
                    revised.append(point)
            return np.array(revised)
        elif self.method == 'Log-FirstDerivatives' or self.method == 7:
            X = np.log10(X)
            revised = []
            for num, point in enumerate(X):
                if num + 1 >= np.shape(X)[1]:
                    revised.append((point[num + 1] - point[num]) / (X_label[num + 1] - X_label[num]))
                else:
                    revised.append(point)
            return np.array(revised)
        elif self.method == 'Log-Inverse-FirstDerivatives' or self.method == 8:
            X = np.reciprocal(np.log10(X))
            revised = []
            for num, point in enumerate(X):
                if num + 1 >= np.shape(X)[1]:
                    revised.append((point[num + 1] - point[num]) / (X_label[num + 1] - X_label[num]))
                else:
                    revised.append(point)
            return np.array(revised)
        elif self.method == 'Sqrt-FirstDerivatives' or self.method == 9:
            X = np.sqrt(X)
            revised = []
            for num, point in enumerate(X):
                if num + 1 >= np.shape(X)[1]:
                    revised.append((point[num + 1] - point[num]) / (X_label[num + 1] - X_label[num]))
                else:
                    revised.append(point)
            return np.array(revised)
        elif self.method == 'Sqrt-Inverse' or self.method == 10:
            return np.reciprocal(np.sqrt(X))
        elif self.method == 'Sqrt-Inverse-FirstDerivatives' or self.method == 11:
            X = np.reciprocal(np.sqrt(X))
            revised = []
            for num, point in enumerate(X):
                if num + 1 >= np.shape(X)[1]:
                    revised.append((point[num + 1] - point[num]) / (X_label[num + 1] - X_label[num]))
                else:
                    revised.append(point)
            return np.array(revised)


from sklearn.linear_model import LinearRegression


class MultipleScatterCorrection(BaseEstimator, TransformerMixin):
    def __init__(self, iterations):
        self.iterations = iterations
        self.M = []

    def fit(self, X, y=None):
        for iters in range(self.iterations):
            X = np.array(X)
            size = np.shape(X)
            _M = np.mean(X, axis=0)
            self.M.append(_M)
            for i in range(size[0]):
                reg = LinearRegression()
                reg.fit(self.M[iters].reshape(-1, 1), X[i].reshape(-1, 1))
                k = reg.coef_[0][0]
                b = reg.intercept_[0]
                X[i] = (X[i] - b) / k
        return self

    def transform(self, X, y=None):
        for iters in range(self.iterations):
            X = np.array(X)
            size = np.shape(X)
            for i in range(size[0]):
                reg = LinearRegression()
                reg.fit(self.M[iters].reshape(-1, 1), X[i].reshape(-1, 1))
                k = reg.coef_[0][0]
                b = reg.intercept_[0]
                X[i] = (X[i] - b) / k
        return X


from scipy.signal import savgol_filter


class SavitzkyGolayFilter(BaseEstimator, TransformerMixin):
    def __init__(self, window_length=5, polyorder=2, deriv=0, delta=1.0, axis=-1, mode='interp', cval=0.0):
        self.window_length = window_length
        self.polyorder = polyorder
        self.deriv = deriv
        self.delta = delta
        self.axis = axis
        self.mode = mode
        self.cval = cval

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        X = savgol_filter(X, self.window_length, self.polyorder, self.deriv, self.delta, self.axis, self.mode,
                          self.cval)
        return X


from pywt import dwt


class HarrDwt(BaseEstimator, TransformerMixin):
    def __init__(self, wavelet='haar', mode='symmetric', axis=-1, iterations=None):
        self.wavelet = wavelet
        self.mode = mode
        self.axis = axis
        self.iterations = iterations

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        if self.iterations == None:
            X, D = dwt(X, self.wavelet, self.mode, self.axis)
        else:
            for i in range(self.iterations):
                X, D = dwt(X, self.wavelet, self.mode, self.axis)
        return X
