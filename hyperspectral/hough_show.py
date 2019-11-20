import cv2
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter.filedialog import askdirectory
import os
from skimage import exposure, img_as_float, img_as_ubyte
import time
import math

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
plt_line = 3
plt_row = 1

'''
plt initialize
'''
plt.figure('hyperspectral image', figsize=(19.2, 10.8), dpi=100)
plt.suptitle('Hyperspectral Image', fontsize=40)

'''
ACTION:
    read data form csv file
    Hough Cirles recognize sample cell
    plt draw picture and save to file
    circle's characteristic save to file
'''
num = 1
for item in csv_file[0:1]:
    print(item)
    with open(path + '/' + item) as f:

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
        # print(color_img)
        # color_img = img_as_float(color_img)
        # print(color_img)
        color_img = color_img / 256
        color_img = color_img.astype(np.uint8)
        color_img = img_as_ubyte(color_img)

        '''
        from csv-file read gray image
        '''
        gray_img = df.query('band == "169"')
        gray_img = img_as_float(gray_img)
        gray_img = exposure.rescale_intensity(gray_img)
        gray_img = gray_img * 256
        while exposure.is_low_contrast(gray_img):
            gray_img = exposure.adjust_gamma(gray_img, 0.8)
        gray_img = gray_img.astype(np.uint8)
        gray_img = img_as_ubyte(gray_img)
        '''cv2 median filter-->>Reduce Noise'''
        '''Optional:cv2 gaussian filter-->>cv2.GaussianBlur'''
        gray_img = cv2.medianBlur(gray_img, 3)
        '''cv2 threshold-->>gain contrast'''
        '''Optional:cv2.adaptiveThreshold-->>have no good result'''
        ret, gray_img = cv2.threshold(gray_img, 50, 255, cv2.THRESH_BINARY)

        '''
        Hough Circles recognize sample cell
        '''
        circles1 = cv2.HoughCircles(gray_img, cv2.HOUGH_GRADIENT, 2, 100, param1=100, param2=25, minRadius=230,
                                    maxRadius=280)

        '''
        HoughLinesP recognize white borad
        load gray image & transfrom with canny
        Draw recongnized lines
        Draw central rectangle
        '''
        cannyimg = df.query('band == "17"')
        cannyimg = np.array(cannyimg)
        cannyimg = img_as_float(cannyimg)
        cannyimg = exposure.rescale_intensity(cannyimg)
        cannyimg = cannyimg * 256
        while exposure.is_low_contrast(cannyimg):
            cannyimg = exposure.adjust_gamma(cannyimg, 0.8)
        cannyimg = cannyimg.astype(np.uint8)
        cannyimg = img_as_ubyte(cannyimg)
        '''cv2 median filter-->>Reduce Noise'''
        '''Optional:cv2 gaussian filter-->>cv2.GaussianBlur'''
        # cannyimg = cv2.medianBlur(cannyimg, 3)
        cannyimg = cv2.GaussianBlur(cannyimg, (5, 5), 0)
        # cannyimg = cv2.fastNlMeansDenoising(cannyimg,h=3,templateWindowSize=7,searchWindowSize=21)
        '''cv2 threshold-->>gain contrast'''
        '''Optional:cv2.adaptiveThreshold-->>have no good result'''
        ret, cannyimg = cv2.threshold(cannyimg, 20, 255, cv2.THRESH_BINARY)
        '''cv2 canny-->>find high contrast area'''
        cannyimg = cv2.Canny(cannyimg, 90, 25)
        '''cv2 HoughLinesP-->>find lines'''
        lines = cv2.HoughLinesP(cannyimg, 1, np.pi / 360, 50, minLineLength=100, maxLineGap=30)


        def Linear_X(y, b, k):
            x = (y - b) / k
            return int(x)


        rectangle_point = []
        try:
            len(lines)
        except TypeError:
            print('no lines')
        else:
            for i in lines:
                for x1, y1, x2, y2 in i:
                    if abs((y1 - y2) / (x1 - x2 + 1E-04)) > 7:
                        cv2.line(color_img, (x1, y1), (x2, y2), (0, 255, 255), 5)
                        K = (y1 - y2) / (x1 - x2 + 1E-04)
                        if K < 0:
                            B = (x2 * y1 - y2 * x1) / (x2 - x1 + 1E-04)
                            X = Linear_X(0, B, K)
                            print(B)
                            rectangle_point.append([B, K,X])
        rectangle_point = np.array(rectangle_point)
        rectangle_point_index = np.where(rectangle_point == np.min(rectangle_point[:, 2]))
        rectangle_point = rectangle_point[rectangle_point_index[0][0]]
        print(rectangle_point)
        B = int(rectangle_point[0])
        K = int(rectangle_point[1])

        '''draw parallelogram'''
        color_array = (255, 255, 0)
        line_weight = 5
        top_point = 175
        bot_point = 675
        L_offset = -135
        R_offset = -35
        cv2.line(color_img,
                 ((Linear_X(top_point, B, K) + L_offset), top_point),
                 ((Linear_X(bot_point, B, K) + L_offset), bot_point),
                 color_array, line_weight)
        cv2.line(color_img,
                 ((Linear_X(top_point, B, K) + R_offset), top_point),
                 ((Linear_X(bot_point, B, K) + R_offset), bot_point),
                 color_array, line_weight)
        cv2.line(color_img,
                 ((Linear_X(top_point, B, K) + L_offset), top_point),
                 ((Linear_X(top_point, B, K) + R_offset), top_point),
                 color_array, line_weight)
        cv2.line(color_img,
                 ((Linear_X(bot_point, B, K) + L_offset), bot_point),
                 ((Linear_X(bot_point, B, K) + R_offset), bot_point),
                 color_array, line_weight)

        '''
        Exception Handling
            hough circles NOT identify circles
        Exception:
            TypeError
        '''
        try:
            circles = circles1[0, :, :]
        except TypeError:
            pass
            print('no circles')
        else:
            circles = np.uint16(np.around(circles))
            '''
            judge circles' position
                i[0]-->> X > 480:
                select circles in right half in the picture

            '''
            temp = circles
            if np.shape(temp)[0] == 1:
                circle = temp[0]
            else:
                countarray = []
                for i in temp:
                    sumarray = []
                    x = i[0]
                    y = i[1]
                    r = i[2]
                    for yi, xline in enumerate(gray_img):
                        for xi, value in enumerate(xline):
                            left = ((x - xi) ** 2 + (y - yi) ** 2)
                            right = r ** 2
                            if left <= right:
                                sumarray.append(value)
                    sumarray = np.array(sumarray)
                    sum = np.mean(sumarray)
                    countarray.append(sum)
                countarray = np.array(countarray)
                print(countarray)
                index = np.where(countarray == np.min(countarray))
                # temp = np.uint16(temp)
                circle = temp[index[0][0]]
            print(circle[0], circle[1], circle[2])
            cv2.circle(color_img, (circle[0], circle[1]), circle[2], (0, 255, 0), 5)
            cv2.circle(color_img, (circle[0], circle[1]), 150, (0, 0, 255), 5)
            cv2.circle(color_img, (circle[0], circle[1]), 6, (255, 0, 0), -1)

            while True:
                item = os.path.splitext(item)
                if item[1] == '':
                    item = item[0]
                    break
                item = item[0]
            '''plt draw picture'''

            '''plt draw origin picture'''
            plt.subplot(plt_row, plt_line, num)
            plt.imshow(color_img)
            plt.title(item, verticalalignment='baseline',
                      horizontalalignment='center', fontsize=10)
            plt.xticks([]), plt.yticks([])
            num = num + 1

            '''plt draw canny picture'''
            plt.subplot(plt_row, plt_line, num)
            plt.imshow(cannyimg, cmap='gray')
            plt.title(item + ' Canny', verticalalignment='baseline',
                      horizontalalignment='center', fontsize=10)
            plt.xticks([]), plt.yticks([])
            num = num + 1

            '''plt draw gray picture'''
            plt.subplot(plt_row, plt_line, num)
            plt.imshow(gray_img, cmap='gray')
            plt.title(item + ' Gray', verticalalignment='baseline',
                      horizontalalignment='center', fontsize=10)
            plt.xticks([]), plt.yticks([])

plt.axis('off')
end_time = time.time()
tol_time = end_time - start_time
print('total time = ' + str(math.ceil(tol_time)) + 's')

'''
plt save picture
'''
plt.savefig('D:\\Desktop\\hough_show.jpg', dpi=400)

plt.show()
