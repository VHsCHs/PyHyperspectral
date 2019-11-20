from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os
import time
import tkinter as tk
from tkinter.filedialog import askdirectory
from private_model.pymodel import nine_point_average

'''
ACTION:
    tkinter initialize
RETURN:
    path ---- select path
    dirs ---- file under path
    csv_file ---- csv-file searched in dirs
'''


def selectpath():
    path_ = askdirectory()
    path.set(path_)


def enter():
    window.destroy()


window = tk.Tk()
path = tk.StringVar()
window.title('select data directory')
window.geometry('400x100')
label = tk.Label(window, text='目标路径：').grid(row=0, column=0)
entry = tk.Entry(window, textvariable=path).grid(row=0, column=1)
select = tk.Button(window, text='select path', command=selectpath).grid(row=1, column=0)
enter = tk.Button(window, text='enter', command=enter).grid(row=1, column=1)
window.mainloop()
path = path.get()
print(path)
dirs = os.listdir(path)
print(dirs)
csv_file = []
for i in dirs:
    if os.path.splitext(i)[1] == ".csv":
        csv_file.append(i)
print(csv_file)

group = []
for i in csv_file:
    i = os.path.splitext(i)[0]
    group.append(i.split('-')[0])
group = list(set(group))
group.sort()
print(group)

class_dict = {}
for i in group:
    for j in dirs:
        if j.split('-')[0] == i:
            class_dict.setdefault(i, []).append(j)
print(class_dict)

project_time = time.strftime("%F-%H-%M-%S")
if os.path.exists(os.path.abspath(os.path.join(os.path.join(path, '..'), '..')) + '\\curves\\') == False:
    os.mkdir(os.path.abspath(os.path.join(os.path.abspath(os.path.join(path, '..')), '..')) + '\\curves\\')
if os.path.exists(os.path.abspath(os.path.join(os.path.join(path, '..'), '..')) + '\\curves\\' + project_time) == False:
    os.mkdir(
        os.path.abspath(os.path.join(os.path.abspath(os.path.join(path, '..')), '..')) + '\\curves\\' + project_time)
# plt_line = 4
# plt_row = math.ceil(len(class_dict) / plt_line)
with open(os.path.split(os.path.split(path)[0])[0] + '/etc/wavelength.csv') as wavelength:
    wl = pd.read_csv(wavelength)
    wl = np.array(wl)
    for num, i in enumerate(class_dict.keys()):
        plt.figure(i,figsize=(9.6,5.4),dpi=100)
        # plt.suptitle(i,fontsize=30)
        plt.axis([400,1000,0,0.6])
        # plt.subplot(plt_row, plt_line, num + 1)
        # plt.title(i)
        for j in class_dict[i]:
            with open(path + '\\' + j, 'r') as f:
                df = pd.read_csv(f)
                df = np.array(df)
                origin = df[:, 1]
                revised = nine_point_average(origin)

                plt.subplot(1,2,1)
                plt.plot(wl, origin, label=j)
                plt.legend(loc='lower right')
                plt.title('origin',fontsize=20)
                plt.subplot(1,2,2)
                plt.plot(wl,revised,label=j)
                plt.legend(loc='lower right')
                plt.title('revised',fontsize=20)
        plt.savefig(os.path.abspath(
            os.path.join(os.path.abspath(os.path.join(path, '..')), '..')) + '/curves/' + project_time + '/' +
                    i + '.jpg', dpi=100)
# plt.show()
