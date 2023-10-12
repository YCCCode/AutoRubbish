'''{
    "0": "A1易拉罐",
    "1": "A2小号矿泉水瓶",
    "2": "A3纸杯",
    "3": "A4铁钉",
    "4": "A5牛奶包装纸盒",
    "5": "A6废纸",
    "6": "B1号电池",
    "7": "B2号电池",
    "8": "B3号电池",
    "9": "B4纽扣电池",
    "10": "B5彩笔",
    "11": "B6药品内包装",
    "12": "C1橘子皮",
    "13": "C2切后菜花",
    "14": "C3切后西兰花",
    "15": "C4青辣椒",
    "16": "C5红辣椒",
    "17": "C6茶叶",
    "18": "C7花椒",
    "19": "C8玉米块",
    "20": "D1碎瓷片",
    "21": "D2烟头",
    "22": "D3口罩",
    "23": "D4竹筷",
    "24": "D5小餐盒",
    "25": "D6牙刷"
}'''


from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QVBoxLayout, QPushButton, QLabel, QFileDialog
from PyQt5.QtCore import QUrl
import sys
import serial
import cv2
import torch
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import json
import time

cur_time = time.time()
from torchvision.models import MobileNetV2

print(torch.__version__)

data_transform = transforms.Compose(
    [transforms.Resize(256),
     transforms.CenterCrop(224),
     transforms.ToTensor(),
     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
i = 0

cap = cv2.VideoCapture(0)

while(i<100):
    i+=1
    ret, frame = cap.read()
    cv2.imwrite('/home/ycc/projects/garbage/test/1.jpg', frame)

    img = Image.open("/home/ycc/projects/garbage/test/1.jpg")


    # [N, C, H, W]
    img = data_transform(img)
    # expand batch dimension
    img = torch.unsqueeze(img, dim=0)
    # print("img:", img)

    try:
        json_file = open('./class_indices.json', 'r')
        class_indict = json.load(json_file)
    except Exception as e:
        print(e)
        exit(-1)

    # create model
    model = MobileNetV2(num_classes=26)
    # load model weights
    model_weight_path = "../weight/mobilenet_v2_Ori_2.6.pth"
    model.load_state_dict(torch.load(model_weight_path, map_location=torch.device('cpu')))
    # torch.load(model_weight_path)
    model.eval()

    with torch.no_grad():
        # predict class
        output = torch.squeeze(model(img))
        predict = torch.softmax(output, dim=0)
        predict_cla = torch.argmax(predict).numpy()
    print(predict_cla)

cap.release()

