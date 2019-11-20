from sklearn.utils.validation import column_or_1d
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import r2_score

file = 'D:\\Desktop\\datasave_without_smooth.csv'
path = 'D:\\Desktop\\'
with open(file, 'r') as f:
    data = pd.read_csv(f, index_col=[0], header=[0])

data = data.query('moisture == 0')
x = data.loc[:, '404.7':'1010.8']
x = np.array(x)
y = data.loc[:, 'TN':'TN']
print(y)
y = np.array(y)
y = y.ravel()
y = column_or_1d(y, warn=True)

print(x)
print(y)

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=1)

forest = RandomForestRegressor(n_estimators=200, criterion='mse', max_depth=9,
                          min_samples_split=2, min_samples_leaf=1,
                          min_weight_fraction_leaf=0.0, max_features=1,
                          max_leaf_nodes=None, min_impurity_decrease=0.0,
                          min_impurity_split=None, bootstrap=True, oob_score=True,
                          n_jobs=None, random_state=1, verbose=0, warm_start=False)
MLR = MLPRegressor(
    solver='lbfgs', activation='relu', alpha=0.0001, hidden_layer_sizes=(500, 50), random_state=1, max_iter=1000000,
    learning_rate_init=0.0001, learning_rate='constant', tol=1e-4, power_t=0.5, shuffle=True,
    verbose=False, warm_start=False, momentum=0.9, nesterovs_momentum=True, early_stopping=False, beta_1=0.9,
    beta_2=0.999, epsilon=1e-08)

MLR.fit(x_train, y_train)
plt.figure('regressor', figsize=(4.8, 5.4), dpi=200)
plt.subplot(2, 1, 1)
plt.title('MLPRegressor')
plt.scatter(x_test[:, 0], y_test, label='origin', c='b', s=10)
plt.scatter(x_test[:, 0], MLR.predict(x_test), label='predict', c='r', s=3)
plt.text(0.305, 1.5, 'score=' + str(round(r2_score(y_test,MLR.predict(x_test)),3)), fontsize=6)
plt.xticks([]), plt.yticks(fontsize=6)
plt.legend(fontsize=6)

forest.fit(x_train, y_train)
plt.subplot(2, 1, 2)
plt.title('RandomForestRegresoor')
plt.scatter(x_test[:, 0], y_test, label='origin', c='b', s=10)
plt.scatter(x_test[:, 0], forest.predict(x_test), label='predict', c='r', s=3)
plt.text(0.305, 1.5, 'score=' + str(round(r2_score(y_test,forest.predict(x_test)),3)), fontsize=6)
plt.xticks([]), plt.yticks(fontsize=6)
plt.legend(fontsize=6)

plt.savefig(path + 'regressor.jpg')
plt.show()

# # print(x)
# # print(y)
# sel = SelectKBest(f_regression, k=10)
# sel.fit(x, y)
# # print(sel.get_support())
#
# reg1 = LinearRegression()
