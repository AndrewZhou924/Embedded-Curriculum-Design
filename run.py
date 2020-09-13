import serial #导入模块
import time
import threading
import easyocr
import numpy as np
from   PIL import Image
from   cnnRecognition.app.CNNinference import *
from   pix2pix.GAN_inference_warpper   import *
from   quickDraw.quickDrawInference    import *

def run_arg_parse():
    parser = argparse.ArgumentParser(description='code for runing!',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--cnn',  '-cnn',  type=bool, default=False)
    parser.add_argument('--en',   '-en',   type=bool, default=False)
    parser.add_argument('--draw', '-draw', type=bool, default=False)
    parser.add_argument('--gan',  '-gan',  type=bool, default=False)
    args = parser.parse_args()
    
    return args

# UART drive config
STRGLO   = ""     # 读取的临时数据
all_data = ""     # 读取的总数据
BOOL     = True   # 读取标志位
SAVEDATA = False  # 是否存储UART数据
SAVEIMG  = True   # 是否存储IMG图片
scene    = 1      # 1处理串口，2直接处理数据
ARGS     = run_arg_parse()

# AI Algorithm model prepare
CNNgraph, CNNsess = predictPrepare() 
OCRreader         = easyocr.Reader(['en'])
quickDrawNet      = getQDmodel()
print('*'*50)
print('Load AI model sucessfully')

# random responce
cnnSureResponce    = ["好字!", "你写的字也太好了~", "这字, 爱了爱了~"]
cnnUnsureResponce  = ["大兄弟,你这写的是啥?", "这字,你是在难为我胖虎!?", "建议多练练硬笔书法~"]
drawUnsureResponce = ["您就是抽象派?", "抱歉,我实在猜不出来", "这画可真难倒我了"]
drawSureResponce   = ["画的针不绰", "这下我该猜对了吧 O(∩_∩)O"]

#读数代码本体实现
def ReadData(ser):
    global STRGLO,BOOL, all_data, SAVEDATA
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
                filename = time.strftime("%m%d_%H_%M_%S", time.localtime())+'.txt'
                if SAVEDATA:
                    SaveData(all_data,filename)
                createImg(all_data,filename)
                all_data = " "

def SaveData(all_data,fileName):
    #print("---------")
    #print(all_data)
    f = open("./realTimeData/"+fileName, 'w')
    f.write(str(all_data))
    f.close()
    
def createImg(all_data,fileName):
    global SAVEIMG, CNNgraph, CNNsess, OCRreader, quickDrawNet, ARGS
    
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
        savePath = "./realTimeData/" + fileName.replace('.txt', '.jpg')
        im.save(savePath)
        print("\n==> save img to: ", savePath)
        
    '''
        Run AI algorithms
    '''
    # Chinese character recognition
    if ARGS.cnn:
        CNNresult = chineseRecognizeSingleImageWithSess(CNNgraph, CNNsess, savePath)
        top1Acc   = CNNresult['pred1_acc_float']
        if float(top1Acc) > 0.1:
            print("==> CNNresult: ", CNNresult['pred1_cnn'], CNNresult['pred1_accuracy'])
            
            # show combined image
            pred1_image_file = CNNresult['pred1_image']
            pred_img = Image.open(pred1_image_file).resize((480, 480))
            ori_img  = Image.open(savePath)
            target   = Image.new('RGB', (800, 480))
            target.paste(ori_img,  (0,  0,320,480))
            target.paste(pred_img, (320,0,800,480))
            target.show()
            
            if top1Acc > 0.7:
                randomGoodComment = "AI助理(*^__^*): " + cnnSureResponce[np.random.randint(0, len(cnnSureResponce))] 
                print(randomGoodComment)
        
        else:
            randomBadComment = "AI助理(*^__^*): " + cnnUnsureResponce[np.random.randint(0, len(cnnUnsureResponce))]
            print(randomBadComment)
    
    '''
        0: bbox  1: result  2: confidence
        ([[26, 382], [84, 382], [84, 436], [26, 436]], 'i2', 0.8742373585700989)
    '''
    if ARGS.en:
        ocrResult = OCRreader.readtext(savePath)
        for res in ocrResult:
            if res[2] > 0.1:
                print("==> OCR Result: ", res[1])
    
    if ARGS.draw:
        [pred, pred_cls, pred_conf] = QDinference(savePath, net=quickDrawNet)
        if pred_conf > 0.2:
            # print("==> quickDraw Result: ", pred, pred_cls)
            print("==> [你画我猜结果]: {}, 置信度: {:.2%}".format(pred_cls, pred_conf))
            
            if pred_conf > 0.7:
                randomGoodComment = "AI助理(*^__^*): " + drawSureResponce[np.random.randint(0, len(drawSureResponce))]
                print(randomGoodComment)
        else:
            randomBadComment = "AI助理(*^__^*): " + drawUnsureResponce[np.random.randint(0, len(drawUnsureResponce))]
            print(randomBadComment)
    
    if ARGS.gan:
        resultPath = GAN_generate(savePath)
        # resultImg  = Image.open(resultPath)
        # resultImg.show()
        
        pred_img = Image.open(resultPath).resize((480, 480))
        ori_img  = Image.open(savePath)
        target   = Image.new('RGB', (800, 480))
        target.paste(ori_img,  (0,  0,320,480))
        target.paste(pred_img, (320,0,800,480))
        target.show()

def drawPoint(data_np,x,y):
    temp = [[-1,-1],[-1,0],[-1,1],[0,-1],[0,0],[0,1],[1,-1],[1,0],[1,1]]
    for i in range(9):
        if  (x+temp[i][0])>=480 or (x+temp[i][0])<0 or (y+temp[i][1])>=320 or (y+temp[i][1])<0: continue
        data_np[x+temp[i][0], y+temp[i][1], :] = np.array([0, 0, 0])

def DOpenPort(portx,bps,timeout):
    '''
    打开串口
    端口，GNU / Linux上的/ dev / ttyUSB0 等 或 Windows上的 COM3 等
    波特率，标准值之一：50,75,110,134,150,200,300,600,1200,1800,2400,4800,9600,19200,38400,57600,115200
    超时设置,None：永远等待操作，0为立即返回请求结果，其他值为等待超时时间(单位为秒）
    '''
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
    # check folder
    if not os.path.exists('./realTimeData/'):
        os.mkdir('./realTimeData/')
        
    if scene == 1:

        ser,ret = DOpenPort("/dev/ttyUSB0", 62500, None)
        if(ret == True): # 判断串口是否成功打开
            while(1):
                time.sleep(5)

    elif scene == 2:
        f = open('test.txt', 'r')
        all_string = ""
        for line in f.readlines(): 
            all_string += line
        createImg(all_string,'test.txt')