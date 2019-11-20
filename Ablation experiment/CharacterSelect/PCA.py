from sklearn.decomposition import PCA
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from models.pymodel import RMSE
from sklearn.model_selection import train_test_split

file = 'D:\\Desktop\\PyProject\\spectrastar f3010\\save_without_WT.csv'

with open(file, 'r') as f:
    df = pd.read_csv(f)
    # for i in range(df.shape[0]):
    #     if len(df.loc[i, 'soil_num']) == 9:
    #         df.loc[i, 'soil_num'] = df.loc[i, 'soil_num'][0:8]
    # d_rows = df[df['soil_num'].duplicated(keep=False)]
    # df.drop(d_rows.index, axis=0, inplace=True)
    # g_item = d_rows.groupby('soil_num').mean()
    # df = df.append(g_item, sort=False)
    # df.sort_values(by='soil_num', inplace=True, ascending=False)
    # df.drop(['soil_num'], axis=1, inplace=True)
    X_array = np.array(df.loc[:, 'point_1':'point_9'])
    Y_array = np.array(df.loc[:, 'TOC':'TOC'])

col = df.loc[:, 'TOC':'WT']
# col = list(col.columns.values)
# print(col)
fig = plt.figure('PCA', figsize=(7.6, 4.8), dpi=100)
n = 4
for i in range(6):
    ax = fig.add_subplot(2 ,3, i + 1)
    if n == 4:
        ax.set_title('source data',fontsize=10)
    else:
        ax.set_title('n_components =' + str(n),fontsize=10)

    x = X_array
    y = Y_array
    if n > 4:
        pca = PCA(n_components=n, copy=True, whiten=False)
        pca.fit(x, y)
        x = pca.transform(x)

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

    forest = RandomForestRegressor(n_estimators=200,
                                   criterion='mse', max_depth=9,
                                   min_samples_split=2, min_samples_leaf=1,
                                   min_weight_fraction_leaf=0.0, max_features=1,
                                   max_leaf_nodes=None, min_impurity_decrease=0.0,
                                   min_impurity_split=None, bootstrap=True, oob_score=True,
                                   n_jobs=None, random_state=1, verbose=0, warm_start=False)

    # forest = ExtraTreesRegressor(n_estimators=100)
    # forest = MLPRegressor(
    #     hidden_layer_sizes=(5, 3), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
    #     learning_rate='constant', learning_rate_init=0.001, power_t=0.5, max_iter=15000, shuffle=True,
    #     random_state=1, tol=0.0001, verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True,
    #     early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
    forest.fit(x, y)
    ax.scatter(x_test[:, 0], y_test, color='teal', s=10, label='origin')
    ax.scatter(x_test[:, 0], forest.predict(x_test), color='chocolate', s=3, label='predict')
    ax.set_xticks([])
    ax.text(0.54, 0.80,
            'r^2=' + str(round(r2_score(y_test, forest.predict(x_test)), 3)) +
            '\nrmse=' + str(round(RMSE(y_test, forest.predict(x_test)), 3)) , transform=ax.transAxes, fontsize=8,bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="gray", lw=1,alpha=0.5)
            )
    ax.legend(fontsize=8, loc='upper left')
    n = n + 1
# pca = PCA(n_components=2)
# pca = PCA(n_components='mle', copy=True, whiten=False)
# # n_components 保留特征数，‘mle’为自动选择。copy将原始训练数据复制。whiten白化，使得每个特征具有相同方差
# pca.fit()
# pca.fit_transform()
# pca.transform()
# pca.inverse_transform()
# plt.savefig('D:\\Desktop\\PCA.jpg', dpi=400)
plt.show()
