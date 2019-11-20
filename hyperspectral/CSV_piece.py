import pandas as pd
import numpy as np
import os
import csv

path = 'D:\\Desktop\\2019-10-29-23-23-53'
dirs = os.listdir(path)
files = []
for i in dirs:
    if os.path.splitext(i)[1] == '.csv':
        files.append(i)
save = open('D:\\Desktop\\datasave_without_smooth.csv', 'w', newline='')
f_csv = csv.writer(save)

with open('D:\\Desktop\\wavelength.csv', 'r') as f:
    header = pd.read_csv(f, header=[0])
header = np.array(header)
header = header.reshape(1, -1)
header = header[0]
header = header.tolist()
header.insert(0, 'moisture')
header.insert(0, 'soil_num')
for i in list(['TOC', 'PH', 'TN', 'SN', 'SP', 'SK', 'WT']):
    header.insert(len(header), i)
f_csv.writerow(header)

with open('D:\\Desktop\\data.csv', 'r') as f:
    data = pd.read_csv(f)
    # print(data)
    for file in files:
        print(file)
        if len(str.split(str.split(os.path.splitext(file)[0], '_')[0], '-')) == 1:
            moisture = 0
            soil_num = str.split(os.path.splitext(file)[0], '_')[0]
        elif (str.split(str.split(os.path.splitext(file)[0], '_')[0], '-'))[1] in list(['05', '10', '15', '20', '30']):
            moisture = int(str.split(str.split(os.path.splitext(file)[0], '_')[0], '-')[1])
            soil_num = str.split(os.path.splitext(file)[0], '_')[0]
        elif (str.split(str.split(os.path.splitext(file)[0], '_')[0], '-'))[1] == '1':
            moisture = 0
            soil_num = str.split(os.path.splitext(file)[0], '_')[0]
        else:
            print('error')
        with open(path + '\\' + file) as csv_file:
            df = pd.read_csv(csv_file, index_col=[0])
            line = np.array(df)
            line = line.reshape(1, -1)
            line = line.tolist()
            line = line[0]
            line.insert(0, moisture)
            line.insert(0, soil_num)
            # print(line)
            linedata = data.query(
                'soil_num ==' + str(int(str.split(str.split(os.path.splitext(file)[0], '_')[0], '-')[0])))
            linedata = linedata.loc[:, 'TOC':'WT']
            linedata = np.array(linedata)
            linedata = linedata.tolist()
            for i in linedata[0]:
                line.insert(len(line), i)
            f_csv.writerow(line)
