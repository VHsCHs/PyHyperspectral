import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
# from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from private_model.p_tkinter import tk_file
from private_model.pymodel import RMSE

path, file_list = tk_file('.csv')

# ['R_save.csv', 'R_save_divide_WT.csv', 'R_save_without.csv', 'R_save_without_WT.csv', 'R_save_without_WT_with_log.csv', 'save.csv', 'save_without_WT.csv']

# for file in file_list:
with open(path + 'save_without_WT.csv','r') as f:
    df = pd.read_csv(f)
    # print(df.columns.values)
    for i in range(df.shape[0]):
        if len(df.loc[i,'soil_num']) == 9:
            df.loc[i, 'soil_num'] = df.loc[i,'soil_num'][0:8]

    d_rows = df[df['soil_num'].duplicated(keep=False)]
    df.drop(d_rows.index,axis=0,inplace=True)
    g_item = d_rows.groupby('soil_num').mean()
    df = df.append(g_item,sort=False)
    df.sort_values(by='soil_num',inplace=True,ascending=False)
    df.drop(['soil_num'],axis=1,inplace=True)
    # print(df)
    X_array = np.array(df.loc[:,'point_1':'point_9'])
    Y_array = np.array(df.loc[:,'TOC':'WT'])
# print(X_array)
scaler = StandardScaler()
scaler.fit(X_array)
X_train = scaler.transform(X_array)

for i in range(Y_array.shape[1]):
    Y_train = Y_array[:,i]
    rf = RandomForestRegressor(n_estimators=200,
                                criterion='mse', max_depth=9,
                                min_samples_split=2, min_samples_leaf=1,
                                min_weight_fraction_leaf=0.0, max_features=1,
                                max_leaf_nodes=None, min_impurity_decrease=0.0,
                                min_impurity_split=None, bootstrap=True, oob_score=True,
                                n_jobs=None, random_state=1, verbose=0, warm_start=False)
    rf.fit(X_train,Y_train)
    print('R^2 =',rf.score(X_train,Y_train))
    print('RMSE = ',RMSE(Y_train,rf.predict(X_train)))


