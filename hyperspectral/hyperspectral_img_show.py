from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import os, math
import time
import tkinter as tk
from tkinter.filedialog import askdirectory
import multiprocessing as mp
import threading as td
import cv2

start_time = time.time()
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
img_num = np.shape(csv_file)[0]
plt.figure('hyperspectal img',figsize=(19.2,10.8),dpi=100)
plt.title('hyperspectal img')

for num, item in enumerate(csv_file):
    with open(path + '/' + item, 'r') as f:
        print(item)
        df = pd.read_csv(f, header=0, index_col=[0, 1])

        '''
        from csv-file read RGB image
        '''
        R_array = np.array(df.query('band == "73"'))
        R_array = R_array.reshape(-1, 1)
        G_array = np.array(df.query('band == "46"'))
        G_array = G_array.reshape(-1, 1)
        B_array = np.array(df.query('band == "15"'))
        B_array = B_array.reshape(-1, 1)
        color_img = np.hstack([R_array, G_array, B_array])
        color_img = color_img.reshape(1101, 960, 3)
        color_img = color_img * 256
        color_img = color_img.astype(np.uint8)
        cv2.rectangle(color_img, (560, 590), (740, 620), (255, 255, 0))

        '''
        get item name & wirte circles parameter
        '''
        while True:
            item = os.path.splitext(item)
            if item[1] != '':
                item = item[0]
            else:
                break

        plt.subplot(4, 7, num + 1)
        plt.title(item[0])
        plt.imshow(color_img)
        plt.xticks([]), plt.yticks([])

end_time = time.time()
tol_time = str(math.ceil(end_time - start_time))
print('total time = ' + tol_time + 's')
plt.axis('off')
plt.show()
