import tkinter as tk
from tkinter.filedialog import askdirectory
import os


def tk_file(file_type='.csv'):
    '''
    ACTION:
        tkinter initialize
    RETURN:
        path ---- select path
        dirs ---- file under path
        files ---- files searched in dirs
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
    path = path + '/'
    print(path)
    dirs = os.listdir(path)
    print(dirs)
    files = []
    for i in dirs:
        if os.path.splitext(i)[1] == file_type:
            files.append(i)
    print(files)
    return path, files
    files = []
    for i in dirs:
        if os.path.splitext(i)[1] == file_type:
            files.append(i)
    print(files)
    return path, files
