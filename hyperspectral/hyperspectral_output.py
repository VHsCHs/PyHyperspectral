from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askdirectory, askopenfilename
import os,sys
import time
import math
from private_model.pymail import MAIL

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
csv_file = []
for i in dirs:
    if os.path.splitext(i)[1] == ".csv":
        csv_file.append(i)
print(csv_file)

'''
ACTION:
    get parameter
return:
    para_array
'''
para = para.get()
with open(para, 'r') as f:
    parameter = pd.read_csv(f, index_col=[0])
# print(parameter)

'''
ACTION:
    get current time 
format:
    YYYY-MM-DD-hh-mm-ss
'''
Project_Time = time.strftime("%F-%H-%M-%S")

'''
line number and camera parameter
'''
plt_line = 10
plt_row = math.ceil(np.shape(csv_file)[0] * 2 / plt_line)

'''
plt initialize
'''
plt.figure('image', figsize=(19.2, 10.8), dpi=100)
plt.title('image')

if os.path.exists(os.path.abspath(os.path.join(path, '..')) + '\\Output\\') == False:
    os.mkdir(os.path.abspath(os.path.join(path, '..')) + '\\Output\\')
if os.path.exists(os.path.abspath(os.path.join(path, '..')) + '\\Output\\' + Project_Time + '\\') == False:
    os.mkdir(os.path.abspath(os.path.join(path, '..')) + '\\Output\\' + Project_Time + '\\')

'''
ACTION:
    read data form csv file
    Hough Cirles recognize sample cell
    plt draw picture and save to file
    circle's characteristic save to file
'''
num = 1
imgsize = 150
for item in csv_file:
    with open(path + '/' + item, 'r') as f:
        while True:
            item = os.path.splitext(item)
            if item[1] != '':
                item = item[0]
            else:
                break
        para = parameter.loc[item[0]]
        print(item[0])

        df = pd.read_csv(f, header=0, index_col=[0, 1])
        # print(df)

        R_band, G_band, B_band = 73, 46, 15
        X_position, Y_position, Radius = para['X-position'], para['Y-position'], para['Radius']
        '''
        color img
        '''
        # cXmin, cYmin, cXmax, cYmax = X_position - Radius, Y_position - Radius, min(X_position + Radius, 959), min(
        #     Y_position + Radius, 1100)
        # R_array = df.loc[(R_band, cYmin):(R_band, cYmax)]
        # R_array = R_array.loc[:, str(cXmin):str(cXmax)]
        # R_array = np.array(R_array)
        #
        # R_array = R_array.reshape(-1, 1)
        #
        # G_array = df.loc[(G_band, cYmin):(G_band, cYmax)]
        # G_array = G_array.loc[:, str(cXmin):str(cXmax)]
        # G_array = np.array(G_array)
        # G_array = G_array.reshape(-1, 1)
        #
        # B_array = df.loc[(B_band, cYmin):(B_band, cYmax)]
        # B_array = B_array.loc[:, str(cXmin):str(cXmax)]
        # B_array = np.array(B_array)
        # B_array = B_array.reshape(-1, 1)
        # color_img = np.hstack([R_array, G_array, B_array])
        # color_img = color_img.reshape(cYmax - cYmin + 1, cXmax - cXmin + 1, 3)
        # color_img = np.array(color_img)
        #
        # '''
        # clip img
        # '''
        cXmin, cYmin, cXmax, cYmax = X_position - imgsize, Y_position - imgsize, min(X_position + imgsize, 959), min(
            Y_position + imgsize, 1100)
        # R_array = df.loc[(R_band, cYmin):(R_band, cYmax)]
        # R_array = R_array.loc[:, str(cXmin):str(cXmax)]
        # R_array = np.array(R_array)
        # R_array = R_array.reshape(-1, 1)
        #
        # G_array = df.loc[(G_band, cYmin):(G_band, cYmax)]
        # G_array = G_array.loc[:, str(cXmin):str(cXmax)]
        # G_array = np.array(G_array)
        # G_array = G_array.reshape(-1, 1)
        #
        # B_array = df.loc[(B_band, cYmin):(B_band, cYmax)]
        # B_array = B_array.loc[:, str(cXmin):str(cXmax)]
        # B_array = np.array(B_array)
        # B_array = B_array.reshape(-1, 1)
        # clip_img = np.hstack([R_array, G_array, B_array])
        # clip_img = clip_img.reshape(cYmax - cYmin + 1, cXmax - cXmin + 1, 3)
        # clip_img = np.array(clip_img)

        # if type(R_array[0][0]) == np.int64:
        #     color_img = color_img / 256
        #     clip_img = clip_img / 256
        #
        #     color_img = color_img.astype(np.uint8)
        #     clip_img = clip_img.astype(np.uint8)
        # elif type(R_array[0][0]) == np.float64:
        #     color_img = img_as_ubyte(color_img)
        #     clip_img = img_as_ubyte(clip_img)
        # else:
        #     print(type(R_array[0][0]))

        # plt.subplot(plt_row, plt_line, num)
        # plt.imshow(color_img)
        # plt.title(item[0], verticalalignment='baseline',
        #           horizontalalignment='center', fontsize=10)
        # plt.xticks([]), plt.yticks([])
        # num = num + 1
        #
        # plt.subplot(plt_row, plt_line, num)
        # plt.imshow(clip_img)
        # plt.title(item[0] + '-clip', verticalalignment='baseline',
        #           horizontalalignment='center', fontsize=10)
        # plt.xticks([]), plt.yticks([])
        # num = num + 1

        clip_array = df.loc[(0, cYmin):(175, cYmax)]
        clip_array = clip_array.loc[:, str(cXmin):str(cXmax)]
        clip_array = clip_array.mean(level=0)
        clip_array = clip_array.mean(axis=1)
        clip_array.to_csv(
            os.path.abspath(os.path.join(path, '..')) + '\\Output\\' + Project_Time + '\\' + item[0] + '.csv', sep=',',
            header=['intensity'],index=True)

# plt.show()
end_time = time.time()
tol_time = end_time - start_time