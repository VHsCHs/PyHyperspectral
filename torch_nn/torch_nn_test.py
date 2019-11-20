# #!/usr/bin/env python
# # -*- coding: utf-8 -*-
#
# import numpy as np
# import pandas as pd
# from matplotlib import pyplot as plt
# import torch as t
# from torch.autograd import Variable as V
#
#
# def main():
#     plt_init()
#     x_array, y_array, x_label = data_read()
#     x_tensor = t.from_numpy(x_label)
#     print(x_tensor)
#     a = V(t.ones(4, 4), requires_grad=True)
#     b = V(t.zeros(4,4))
#     c = a.add(b)
#     d = c.sum()
#     print(d)
#     d.backward()
#     print(d)
#     c.data.sum(),c.sum()
#
#
# def plt_init():
#     # 支持中文
#     plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
#     plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
#
#
# def data_read(file='D:\\Desktop\\datasave_without_smooth.csv'):
#     with open(file, 'r') as f:
#         data = pd.read_csv(f, index_col=[0], header=[0])
#         x_array = data.loc[:, '404.7': '1010.8']
#         y_array = data.loc[:, 'moisture']
#         x_label = x_array.columns
#         x_array, y_array, x_label = np.array(x_array), np.ravel(np.array(y_array)), np.array(x_label).astype(float)
#         return x_array, y_array, x_label
#
#
# if __name__ == '__main__':
#     main()
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
import torch as t
import torch.nn as nn


def main():
    plt_init()
    x_array, y_array, x_label = data_read()
    x_tensor = t.from_numpy(x_label)
    x_tensor.numpy()
    print(x_tensor)


def plt_init():
    # 支持中文
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号


def torch_init():


def data_read(file='D:\\Desktop\\datasave_without_smooth.csv'):
    with open(file, 'r') as f:
        data = pd.read_csv(f, index_col=[0], header=[0])
        x_array = data.loc[:, '404.7': '1010.8']
        y_array = data.loc[:, 'moisture']
        x_label = x_array.columns
        x_array, y_array, x_label = np.array(x_array), np.ravel(np.array(y_array)), np.array(x_label).astype(float)
        return x_array, y_array, x_label


if __name__ == '__main__':
    main()
