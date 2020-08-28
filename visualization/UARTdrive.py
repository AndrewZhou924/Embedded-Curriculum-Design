import serial #导入模块
import time
import threading
from PIL import Image
import numpy as np

STRGLO=""       #读取的临时数据
all_data = ""   #读取的总数据
BOOL=True       #读取标志位
SAVEDATA = True     #是否存储UART数据
SAVEIMG = True      #是否存储IMG图片
scene = 1       # 1处理串口，2直接处理数据


#读数代码本体实现
def ReadData(ser):
    global STRGLO,BOOL,all_data,SAVEDATA
    # 循环接收数据，此为死循环，可用线程实现
    while BOOL:
        if ser.in_waiting:
            STRGLO = ser.read(ser.in_waiting).hex()
            #print(STRGLO)
            STRGLO = str(STRGLO)
            for i in range(len(STRGLO))[::2]:
                char = STRGLO[i]+STRGLO[i+1]
                all_data = all_data+char+" "
            #all_data = all_data + STRGLO
            #print(STRGLO)
            if "ca ae" in all_data:
                filename = time.strftime("%m%d %H_%M_%S", time.localtime())+'.txt'
                if SAVEDATA:
                    SaveData(all_data,filename)
                createImg(all_data,filename)
                all_data = " "

def SaveData(all_data,fileName):
    #print("---------")
    #print(all_data)
    f = open(".\\data\\txt\\"+fileName, 'w')
    f.write(str(all_data))
    f.close()
    

def createImg(all_data,fileName):
    global SAVEIMG
    all_data = str(all_data)
    all_data = all_data.split('fa ae ')[1]
    split= all_data.split(' ')
    #print(split)
    #print(len(split))
    data_np = np.ones((480, 320, 3))
    pixelnum = int((len(split)/3)-2)
    x_data_start = 0
    y_data_start = pixelnum+2
    color_data_start = 2*(pixelnum+2)
    for i in range(pixelnum)[::2]:
        pixel = split[color_data_start+i]+split[color_data_start+i+1]
        temp_x = int((split[x_data_start+i+1]+split[x_data_start+i]),16)
        temp_y = int((split[y_data_start+i+1]+split[y_data_start+i]),16)
        if (temp_x>=320) or (temp_y>=480):  continue
        if pixel != 'FFFF':
            # print(split[x_data_start+i+1]+split[x_data_start+i]+" "+split[y_data_start+i+1]+split[y_data_start+i])
            drawPoint(data_np,temp_y,temp_x)
            #data_np[temp_x, temp_y, :] = np.array([0, 0, 0])
    for i in range(480):
        for j in range(320):
            if (data_np[i,j, : ] ==  np.array([1, 1, 1])).all():
                data_np[i,j, : ] = np.array([255, 255, 255])
    im = Image.fromarray(np.uint8(data_np))
    if SAVEIMG:
        savePath = fileName.replace('.txt', '.jpg')
        im.save(".\\data\\jpg\\"+savePath)
        print("==> save to: ", savePath)

def drawPoint(data_np,x,y):
    temp = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
    for i in range(9):
        if  (x+temp[i][0])>=480 or (x+temp[i][0])<0 or (y+temp[i][1])>=320 or (y+temp[i][1])<0: continue
        data_np[x+temp[i][0], y+temp[i][1], :] = np.array([0, 0, 0])



#打开串口
# 端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
# 波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
# 超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
def DOpenPort(portx,bps,timeout):
    ret=False
    try:
        # 打开串口，并得到串口对象
        ser = serial.Serial(portx, bps, timeout=timeout)
        #判断是否打开成功
        if(ser.is_open):
           ret=True
           threading.Thread(target=ReadData, args=(ser,)).start()
    except Exception as e:
        print("---异常---：", e)
    return ser,ret



#关闭串口
def DColsePort(ser):
    global BOOL
    BOOL=False
    ser.close()



#写数据
def DWritePort(ser,text):
    result = ser.write(text)  # 写数据
    return result




#读数据
def DReadPort():
    global STRGLO
    str=STRGLO
    STRGLO=""#清空当次读取
    return str



if __name__=="__main__":
    if scene == 1:

        ser,ret=DOpenPort("com8",62500,None)
        if(ret==True):#判断串口是否成功打开

            while(1):
                time.sleep(5)
                #print("读取数据:"+DReadPort()) #读串口数据
                #DColsePort(ser)  #关闭串口

    elif scene ==2:
        f = open('test.txt', 'r')
        all_string = ""
        for line in f.readlines(): 
            all_string += line
        createImg(all_string,'test.txt')