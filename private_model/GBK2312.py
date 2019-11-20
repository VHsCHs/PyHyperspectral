# -*-coding:utf-8-*-
import codecs
def GBK2312(head,body):
    # head = random.randint(0xb0, 0xf7)
    # body = random.randint(0xa1, 0xf9)   # 在head区号为55的那一块最后5个汉字是乱码,为了方便缩减下范围
    val = f'{head:x}{body:x}'
    str = bytes.fromhex(val).decode('gb2312')
    return str
n = 0
f = codecs.open('D:\\Desktop\\data.txt','w','utf-8')
for head in range(0xb0,0xf7):
    for body in range(0xa1,0xf9):
        data = GBK2312(head,body) + "\n"
        f.write(data)
        print(n)
        n+=1
f.close()