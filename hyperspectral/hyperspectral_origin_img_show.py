from matplotlib import pyplot as plt
import pandas as pd
import os
import numpy as np
import time
import cv2
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
import math

'''
ACTION:
    tkinter initialize
RETURN:
    path ---- select path
    dirs ---- file under path
    bmp_file ---- csv-file searched in dirs
'''


def selectpath():
    path_ = askdirectory()
    path.set(path_)


def selectpara():
    para_ = askopenfilename()
    para.set(para_)


def enter():
    window.destroy()


window = tk.Tk()
path = tk.StringVar()
para = tk.StringVar()
window.title('select data directory')
window.geometry('400x100')
label = tk.Label(window, text='目标路径：').grid(row=0, column=0)
entry = tk.Entry(window, textvariable=path).grid(row=0, column=1)
paralabel = tk.Label(window, text='参数文件：').grid(row=1, column=0)
entry = tk.Entry(window, textvariable=para).grid(row=1, column=1)
select = tk.Button(window, text='select path', command=selectpath).grid(row=2, column=0)
selectpara = tk.Button(window, text='select para', command=selectpara).grid(row=2, column=1)
enter = tk.Button(window, text='enter', command=enter).grid(row=2, column=2)
window.mainloop()
path = path.get()
print(path)
dirs = os.listdir(path)
print(dirs)
bmp_file = []
for i in dirs:
    if os.path.splitext(i)[1] == ".bmp":
        bmp_file.append(i)
print(bmp_file)

group = []
for i in bmp_file:
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

'''
ACTION:
    get parameter
return:
    para_array
'''
para = para.get()
with open(para, 'r') as f:
    parameter = pd.read_csv(f, index_col=[0], header=[0])
imgsize = 150

project_time = time.strftime("%F-%H-%M-%S")
if os.path.exists(os.path.abspath(os.path.join(path, '..')) + '\\bmp_circles\\') == False:
    os.mkdir(os.path.abspath(os.path.join(path, '..')) + '\\bmp_circles\\')
if os.path.exists(os.path.abspath(os.path.join(path, '..')) + '\\bmp_circles\\' + project_time) == False:
    os.mkdir(os.path.abspath(os.path.join(path, '..')) + '\\bmp_circles\\' + project_time)

for num, i in enumerate(class_dict.keys()):
    plt.figure(i, figsize=(19.2, 10.8), dpi=100)
    plt.suptitle(i)
    for k, j in enumerate(class_dict[i]):
        item_para = parameter.loc[(os.path.splitext(j)[0] + '_ref')]
        X, Y, R = item_para.loc['X-position'], item_para.loc['Y-position'], item_para.loc['Radius']
        cXmin, cYmin, cXmax, cYmax = X - imgsize, Y - imgsize, min(X + imgsize, 959), min(Y + imgsize, 1100)



        plt_row = 2
        plt_line = math.ceil(len(class_dict[i]) / plt_row)
        plt.subplot(plt_row, plt_line, k + 1)
        plt.title(os.path.splitext(j)[0])

        img = plt.imread(path + '\\' + j)
        cv2.rectangle(img, (cXmin, cYmin), (cXmax, cYmax), (0, 255, 0), 2)

        plt.imshow(img)
        plt.xticks([]), plt.yticks([])

    plt.savefig(os.path.abspath(os.path.join(path, '..')) + '\\bmp_circles\\' + project_time + '/' + i + '.jpg',
                dpi=100)
# plt.show()
