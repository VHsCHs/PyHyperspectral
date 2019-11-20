import os
from urllib import parse
from urllib import request
path = os.path.abspath('.')
fileList=os.listdir(path)
n=0
for i in fileList:
    oldname=path+os.sep+fileList[n]
    newname=parse.unquote(oldname)
    print(newname)
    try:
        os.rename(oldname,newname)
    except FileNotFoundError:
        print(oldname+' not found')
    n+=1
print('Press ENTER to quit.')
