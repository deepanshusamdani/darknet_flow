#!/usr/bin/python3

from darkflow.net.build import TFNet
import cv2

options={"model": "cfg/yolo.cfg", "load": "bin/yolo.weights", "threshold":0.1}

tfnet = TFNet(options)

imgcv=cv2.imread("/home/deepu/Desktop/darknet/data/dog.jpg")
result=tfnet.return_predict(imgcv)
print(result)