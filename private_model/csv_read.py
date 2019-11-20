import csv
import numpy as np
def csv_reader(csv_file):
    with open(csv_file, "r", newline="") as data_read:
        reader = csv.reader(data_read)
        for i, rows in enumerate(reader):
            if i == 0:
                var_name = rows
    with open(csv_file, "r", newline="") as data_read:
        sorted_data = csv.DictReader(data_read, fieldnames=var_name)
        listed_data = list(sorted_data)
    n = 0
    csv_list = np.zeros((len(listed_data)-1,len(var_name)-1),dtype=float)
    for x in listed_data:
        temp_list = []
        if x[var_name[0]] == var_name[0]:
            continue
        else:
            for y in var_name[1:len(var_name)]:
                if x[y] == "-∞":
                    temp_list.append(1)
                else:
                    try:
                        temp_list.append(float(x[y]))
                    except ValueError:
                        temp_list.append(1)
        csv_list[n] = temp_list
        n+=1
    return csv_list

def csv_varname(csv_file):
    with open(csv_file, "r", newline="") as data_read:
        reader = csv.reader(data_read)
        for i, rows in enumerate(reader):
            if i == 0:
                var_name = rows
    return var_name

def csv_samplename(csv_file):
    with open(csv_file, "r", newline="") as data_read:
        reader = csv.reader(data_read)
        temp_array = []
        for i in reader:
            if i[0] == 'soil_num':
                continue
            else:
                temp_array.append(i[0])
    return temp_array

def index_of_str(s1, s2):
    n1 = len(s1)
    n2 = len(s2)
    for i in range(n1 - n2 + 1):
        if s1[i:i + n2] == s2:
            for j in range(n1 - n2 + 1 -i):
                if s1[i+j+1:i+j+1 + n2] == s2:
                    return j+i+1
    else:
        return -1

def csv_average(csv_data,csv_file=None):
    if type(csv_data) == str:
        print(csv_data)
    elif type(csv_data) == np.ndarray:
        with open (csv_file, "r", newline="") as data_read:
            reader = csv.DictReader(data_read)
            temp_array = {}
            for num,line in enumerate(reader):
                temp_array[line['soil_num']] = num

    new_array = {}
    for i,j in temp_array.items():
        find = 0
        for k in new_array.keys():
            if k == i[0:index_of_str(i, '-')]:
                find = 1
                break
            else:
                find = 0
        if find == 0:
            new_array[i[0:index_of_str(i, '-')]] = [j]
        elif find == 1:
            new_array[i[0:index_of_str(i, '-')]].append(j)
    data = []
    for name,row in new_array.items():
        averager = []
        for number in row:
            averager.append(csv_data[number])
        averager = np.array(averager)
        averager = averager.mean(axis=0)
        averager = averager.tolist()
        data.append(averager)
    data = np.array(data)
    return data