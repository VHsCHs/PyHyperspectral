import math
import csv
import tkinter as tk
from tkinter.filedialog import askdirectory
import os
def selectpath():
    path_ = askdirectory()
    path.set(path_)

def enter():
    window.destroy()


window = tk.Tk()
path = tk.StringVar()

window.title('select data directory')
window.geometry('400x100')
label = tk.Label(window,text = '目标路径：').grid(row = 0, column = 0)
entry = tk.Entry(window,textvariable = path).grid(row = 0 , column = 1)
select = tk.Button(window,text = 'select path',command = selectpath).grid(row = 1, column = 0)
enter = tk.Button(window,text = 'enter',command = enter).grid(row = 1,column = 1)
window.mainloop()
path = path.get()
print(path)
dirs = os.listdir(path)
print(dirs)
if os.path.exists(os.path.abspath(os.path.join(path,'..')) + '/CSV/') == False:
    os.mkdir(os.path.abspath(os.path.join(path,'..')) + '/CSV')
txt_file = []
for i in dirs:
    if os.path.splitext(i)[1] == ".txt":
        txt_file.append(i)
print(txt_file)

# input_file = 'D:\\Desktop\\BSQ\\311-30.txt'
# output_file = 'D:\\Desktop\\BSQ\\311-30_transform.csv'

for i in txt_file:
    print(i)
    input_file = path + '/' + i
    output_file = os.path.abspath(os.path.join(path,'..')) + '/CSV/' + i + '.csv'
    output_txt = open(output_file,'w',newline='')
    f_csv = csv.writer(output_txt)

    header = list(range(0,960,1))
    header.insert(0,'band')
    header.insert(1,'Y')
    f_csv.writerow(header)

    Y = 0
    BAND = 0
    for num,line in enumerate(open(input_file,'r')):
        if num < 5:
            continue
        # if line == []:
        #     print('line'+line+'.')
        line = line.lstrip()
        col = line.split()
        if col == []:
            continue
        if Y < 1100:
            if num == 5:
                col.insert(0, str(Y))
            else:
                Y += 1
                col.insert(0, str(Y))
            col.insert(0, str(BAND))
        else:
            Y = 0
            BAND += 1
            col.insert(0, str(Y))
            col.insert(0, str(BAND))
        # print(col)
        f_csv.writerow(col)