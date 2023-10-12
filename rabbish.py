from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QWidget, QGridLayout,QTextEdit, QLabel
from PyQt5.QtCore import QUrl,QTimer,QThread,pyqtSignal
from PyQt5.QtGui import QImage,QPixmap
import sys
import serial
import cv2
import torch
from PIL import Image
from torchvision import transforms
import json
from torchvision.models import MobileNetV2
import time
import numpy as np


maxcontin =3


aoverload=False
boverload=False
coverload=False
doverload=False

UartOK=False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1024)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 768)
data_transform = transforms.Compose(
    [transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
q_image:QImage
first_over:bool =False
check_ready:bool = True
model:MobileNetV2
class_indict:dict
isLjRun:bool =False #正在倒入垃圾
lj_count = [0,0,0,0,0]   #第一个作为序号

HASH1:str
class Garbage(QWidget):
    pass
garbage:Garbage
predict_cla:int=-1
curclass:int = -1
curclass_index  =-1
class_count:int = 0
frame = np.zeros((1,1,1), np.float32)
ser:serial.Serial

sys_ready:bool = False
check_threading:bool =False

set_same_value =0.87 #相似度阈值

#多线程模型预估，使用单线程摄像头预览会卡，程序会卡
class CheckThread(QThread):
    def run(self):
        
        global check_ready
        global class_count,curclass,predict_cla
        global check_threading,maxcontin
        
        single_count=0
        while(True):
            img = Image.fromqimage(q_image)
            img = data_transform(img)
            img = torch.unsqueeze(img, dim=0)

            with torch.no_grad():
                output = torch.squeeze(model(img))
                predict = torch.softmax(output, dim=0)
                predict_cla = torch.argmax(predict).numpy()

            garbage.Rc_log('识别为：'+class_indict[str(predict_cla)])
            
            single_count+=1
            if(single_count>=12):
                maxcontin=maxcontin-1
            if(curclass == predict_cla):
                class_count+=1
            else:
                curclass=predict_cla
                class_count=0
            
            if(class_count>=maxcontin):  #***********识别阈值
                garbage.Rc_log('最终识别为：'+class_indict[str(predict_cla)])
                class_count=0
                global curclass_index
                if(curclass<=5):
                    HandlRabbish(1)
                    curclass_index=1
                elif(curclass<=11):
                    HandlRabbish(2)
                    curclass_index=2
                elif(curclass<=19):
                    HandlRabbish(3)
                    curclass_index=3
                elif(curclass<=25):
                    HandlRabbish(4)
                    curclass_index=4
                check_ready=True
                maxcontin=3
                break
            
        
        
thread = CheckThread()

#处理垃圾
def HandlRabbish(class_index:int):
    global isLjRun
    
    if(UartOK==False):
        return
    
    isLjRun = True
    
    lj_count[class_index]+=1
    lj_count[0]+=1
    
    
    
    if(class_index==1):
        garbage.Rc_log('可回收垃圾,开始倒入')
        ser.write(b'1')
    elif(class_index==2):
        garbage.Rc_log('有害垃圾,开始倒入')
        
        ser.write(b'2')
    elif(class_index==3):
        garbage.Rc_log('厨余垃圾,开始倒入')
        ser.write(b'3')
    elif(class_index==4):
        garbage.Rc_log('其余垃圾,开始倒入')
        ser.write(b'4')


class Garbage(QWidget):
    
    
    #涉及多线程更新UI，线程中直接更新UI会导致程序崩溃，所以线程发送更新UI信号，主程序更新UI
    updated_log = pyqtSignal(str)
    
    def __init__(self):
        super().__init__()
        self.showMaximized()
        
        layout = QGridLayout(self)
        
        
        self.setWindowTitle("工训赛-智能分类垃圾桶-----by:ycc")
        self.setGeometry(110, 110, 800, 600)

        self.video_widget = QVideoWidget(self)
        self.video_widget.setGeometry(100,200,500,230)


        self.media_player = QMediaPlayer(self)
        self.media_player.setVideoOutput(self.video_widget)
        self.media_player.stateChanged.connect(self.mediaStateChanged)
        self.open_video()

        self.status_label = QLabel("当前状态:序号  垃圾类别  数量  分类成功与否", self)
        self.status_label.setStyleSheet("font-size: 20px;")
        
        
        self.uart_label = QLabel("串口未连接", self)
        self.uart_label.setStyleSheet("font-size: 10px;")
        
        self.rec_rate_label = QLabel("   相似度:", self)
        self.rec_rate_label.setStyleSheet("font-size: 20px;")
        layout.addWidget(self.rec_rate_label, 2, 2) 
        
        
        self.overload_label = QLabel("无满载", self)
        self.overload_label.setStyleSheet("font-size: 20px;")
        
        self.atrash_label = QLabel("可回收垃圾数量:0", self)
        self.atrash_label.setStyleSheet("font-size: 20px;")

        self.btrash_label = QLabel("厨余垃圾数量:0", self)
        self.btrash_label.setStyleSheet("font-size: 20px;")

        self.ctrash_label = QLabel("有害垃圾数量:0", self)
        self.ctrash_label.setStyleSheet("font-size: 20px;")

        self.dtrash_label = QLabel("其余垃圾数量:0", self)
        self.dtrash_label.setStyleSheet("font-size: 20px;")
        
        self.video_show_label = QLabel("串口连接成功串口连接成功串口", self)
        self.video_show_label.setStyleSheet("font-size: 20px;")
        
        self.log_edit = QTextEdit(self)
        layout.addWidget(self.log_edit,4,2,2,1)
        
        
        layout.addWidget(self.video_show_label, 0, 2,2,2)  

        
        layout.addWidget(self.status_label, 0, 0,1,1)  
        layout.addWidget(self.uart_label, 0, 1)  
        
        layout.addWidget(self.video_widget, 1, 0, 5, 1) 
    
        layout.addWidget(self.overload_label, 1, 1)  
        layout.addWidget(self.atrash_label, 2, 1) 
        
        layout.addWidget(self.btrash_label, 3, 1) 
        
        layout.addWidget(self.ctrash_label, 4, 1)  
        
        layout.addWidget(self.dtrash_label, 5, 1)  
        
        self.initModel()
        
        view_timer = QTimer(self)
        view_timer.timeout.connect(self.view_get)
        view_timer.start(30)
        
        check_rabbish = QTimer(self)
        check_rabbish.timeout.connect(self.checkRabbish)
        check_rabbish.start(300)
        
        rec_uart = QTimer(self)
        rec_uart.timeout.connect(self.RecUart)
        rec_uart.start(1000)
        
        self.updated_log.connect(self.UpdateLog)
        
        try:
            global ser,UartOK
            ser = serial.Serial('/dev/ttyUSB0',9600,timeout = 0.5)
            self.ChangeUartLabal("串口连接成功")
            UartOK=True
        except:
            self.ChangeUartLabal("串口连接失败")
            
        
        self.Rc_log('开机成功,开始检测垃圾...')
        global sys_ready
        
        sys_ready = True
        
        
    
    
    #求哈希
    def pHash(self):
        # 求哈希值的一个东西
      
        img_list = []
        # 加载并调整图片为32x32灰度图片
        #img = Image.fromqimage(q_image)
        #img=cv2.imread('./test/1.jpg',0)
        img = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
        
        img = cv2.resize(img, (32, 32), interpolation=cv2.INTER_CUBIC)
        # 创建二维列表
        h, w = img.shape[:2]
        vis0 = np.zeros((h, w), np.float32)
        
        vis0[:h, :w] = img  # 填充数据
        # 二维Dct变换
        vis1 = cv2.dct(cv2.dct(vis0))
        vis1.resize(32, 32)
        # 把二维list变成一维list
        img_list = vis1.flatten()
        # 计算均值
        avg = sum(img_list) * 1. / len(img_list)
        avg_list = ['0' if i > avg else '1' for i in img_list]
        # 得到哈希值
        return ''.join(['%x' % int(''.join(avg_list[x:x + 4]), 2) for x in range(0, 32 * 32, 4)])

    # 计算相似度
    def hammingDist(self, s1, s2):
        # 求出相似指数的函数

        # assert len(s1) == len(s2)
        return 1 - sum([ch1 != ch2 for ch1, ch2 in zip(s1, s2)]) * 1. / (32 * 32 / 4)

    # 打印并返回相似度
    def hash_score(self):
        HASH2 = self.pHash()
        #print(HASH2)
        out_score = self.hammingDist(HASH1, HASH2)
        #print('相似度：', out_score)
        self.rec_rate_label.setText("   相似度:"+str(round(out_score,3)))
        
        return out_score

    
    
    #接收窜口消息
    def RecUart(self):
        global isLjRun,aoverload,boverload,coverload,doverload,HASH1
        
        if(UartOK==False):
            return
        
        if(self.media_player.position()>68000):
            self.media_player.setPosition(0)
            self.Rc_log('视频重新播放')
        
        
        ms = ser.read_all()
        if(ms!=None and len(ms)>0):
            self.Rc_log('来自窜口消息：'+str(ms))
            
            if(ms==b'0'):
                self.Rc_log('托盘回正,等待投放垃圾...')
                HASH1 = self.pHash()
                isLjRun=False
            else:
                if(self.overload_label.text()=='无满载'):
                    self.overload_label.setText('')
                if(ms==b'\x01'):
                    if(aoverload==False):
                        aoverload=True
                        self.overload_label.setText(self.overload_label.text()+'可回收满载\n')
                elif(ms==b'\x02'):
                    if(boverload==False):
                        boverload=True
                        self.overload_label.setText(self.overload_label.text()+'有害垃圾满载\n')
                elif(ms==b'\x03'):
                    if(coverload==False):
                        coverload=True
                        self.overload_label.setText(self.overload_label.text()+'厨余垃圾满载\n')
                elif(ms==b'\x04'):
                    if(doverload==False):
                        doverload=True
                        self.overload_label.setText(self.overload_label.text()+'其余垃圾满载\n')
                else:
                    self.Rc_log('错误指令')
            
            
    #刷新信息UI
    def RushState(self):
        t=''
        if(curclass_index==1):
            t='可回收垃圾'
        elif(curclass_index==2):
            t='有害垃圾  '
        elif(curclass_index==3):
            t='厨余垃圾  '
        elif(curclass_index==4):
            t='其余垃圾  '
        if(lj_count[0]==0):
            self.status_label.setText('待投入垃圾')
        else:
            if(isLjRun):
                self.status_label.setText('序号'+str(lj_count[0])+'  垃圾类别'+t+'  数量1  分类成功')
            else:
                self.status_label.setText('待投入垃圾')
        self.atrash_label.setText('可回收垃圾数量:'+str(lj_count[1]))
        self.btrash_label.setText('有害垃圾数量:'+str(lj_count[2]))
        self.ctrash_label.setText('厨余垃圾数量:'+str(lj_count[3]))
        self.dtrash_label.setText('其余垃圾数量:'+str(lj_count[4]))
        
    #初始化模型
    def initModel(self):
        global model,class_indict
        try:
            json_file = open('./class_indices.json', 'r')
            class_indict = json.load(json_file)
        except Exception as e:
            print(e)
            self.Rc_log('分类json文件加载错误')
            return

        # create model
        model = MobileNetV2(num_classes=26)
        # load model weights
        model_weight_path = "./mobilenet_v2_Ori_2.6.pth"
        model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
        # torch.load(model_weight_path)
        model.eval()
         
    #发送更新日志信号
    def Rc_log(self,ms:str):
        self.updated_log.emit(ms)
    
    #更新日志UI
    def UpdateLog(self,ms:str):
        self.log_edit.append(ms)
        self.RushState()
    
    #改变窜口标签
    def ChangeUartLabal(self,newstr:str):
        self.uart_label.setText(newstr)
        
    #摄像头预览
    def view_get(self):
        global frame,first_over,HASH1,q_image
        
        
        ret,frame = cap.read()
       
       # 获取输入图像的宽度和高度
        height, width, _ = frame.shape

        # 计算裁剪的起始位置
        start_x = (width - 768) // 2
        end_x = start_x + 768

        # 裁剪图像
        frame = frame[:, start_x:end_x, :]
       
       
        if(first_over==False):
            
            first_over=True
            HASH1 = self.pHash()

        
        rgb_image = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
        
        h,w,ch = rgb_image.shape
        byters_per_line = ch*w
        
        q_image = QImage(rgb_image.data,w,h,byters_per_line,QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        
        self.video_show_label.setPixmap(pixmap.scaled(self.video_show_label.size(),))
        self.video_show_label.setFixedWidth(int(self.video_show_label.height()))
        first_over = True
        
    #播放视频
    def open_video(self):
        file_path = './demo.mp4'
        if file_path:
            media_content = QMediaContent(QUrl.fromLocalFile(file_path))
            self.media_player.setMedia(media_content)
            self.media_player.play()
            
    #视频播放结束重新播放
    def mediaStateChanged(self, state):
        if state == 0:
            self.media_player.setPosition(0)
            self.media_player.play()
            self.Rc_log('重新播放')

    #垃圾检测
    def checkRabbish(self):
        global check_ready,isLjRun,curclass,curclass,class_count
        
        
        #托盘是否准备好  系统是否准备好
        if (check_ready==True )and (isLjRun==False) and sys_ready==True and first_over==True:
            
            cursame = self.hash_score()
            #self.Rc_log('相似度:'+str(cursame))
            if(cursame>set_same_value):
                return
            self.Rc_log('检测到垃圾！开始分类')
            #检测托盘有无垃圾
           
            
            # return
            
            check_ready=False
            thread.start()
               
                
        


if __name__ == "__main__":
    app = QApplication(sys.argv)
    garbage = Garbage()
    garbage.show()
    sys.exit(app.exec())
