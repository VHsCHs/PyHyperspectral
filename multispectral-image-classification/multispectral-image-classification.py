from matplotlib import pyplot as plt
from matplotlib import image as mpimg
import numpy as np
import pandas as pd
import os
import random
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split

dir = 'D:\\Desktop\\dataset\\multispectral-image-classification\\MultiSpectralImages\\'
files = os.listdir(dir)
for num,i in enumerate(files):
    if i == 'Labels.csv':
        # print(i)
        files.pop(num)
    elif i == 'ba295b7e-7893-4ae3-a2f7-54c5f0223e45_uofa.csv':
        # print(i)
        files.pop(num)

files = random.sample(files,10)

def standard_zero_to_one(array):
    mean = array.mean()         #计算平均数
    deviation = array.std()     #计算标准差
    # 标准化数据的公式: (数据值 - 平均数) / 标准差
    standard = np.array((array - mean) / deviation)
    standard = standard + np.abs(standard.min())
    standard = standard / standard.max()
    return standard

X_data = {}
for file in files:
    # print(file)
    img = pd.read_csv(dir + file,usecols=['Channel0','Channel1','Channel2','Channel3','Channel4','Channel5','Channel6','Channel7','Channel8','Channel9'])
    img = np.array(img)
    img = img.flatten()
    img = standard_zero_to_one(img)
    X_data[file] = img

Y_data = {}
temp = pd.read_csv(dir + 'Labels.csv',usecols=['Label','FileName'])
temp = np.array(temp)

for file in files:
    for line in temp:
        if line[1] == file:
            Y_data[file] = line[0]

X_array = []
Y_array = []

for i in X_data:
    for j in Y_data:
        if i == j:
            X_array.append(X_data[i])
            Y_array.append(Y_data[i])
X_array = np.array(X_array)
Y_array = np.array(Y_array)
print(np.shape(X_array))
X_train,X_test,Y_train,Y_test = train_test_split(X_array,Y_array,train_size=0.25, random_state=1)

clf = MLPClassifier(hidden_layer_sizes=(500,30), activation='relu', solver='adam', alpha=0.0001, batch_size='auto',
                    learning_rate='constant', learning_rate_init=0.00000001, power_t=0.5, max_iter=100000, shuffle=True,
                    random_state=1, tol=1e-4, verbose=True, warm_start=False, momentum=0.9, nesterovs_momentum=True,
                    early_stopping=False, beta_1=0.9, beta_2=0.999, epsilon=1e-08)
clf.fit(X_train,Y_train)
prediction = clf.predict(X_test)
temp = 0
for num,i in prediction:
    if i == Y_test[num]:
        temp+=1
print(temp/num)
print(clf.score(X_test,Y_test))