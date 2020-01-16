# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:15:10 2020

@author: Prasan
"""


import urllib
import cv2
import numpy as np
import pickle
from visualize import visualize_images


classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

URL = "http://192.168.42.129:8080/shot.jpg"

def preprocess(img):
    img = cv2.resize(img,(200,200))
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return img

data = []
ret = True
while ret:
    img = urllib.request.urlopen(URL)
    image = np.array(bytearray(img.read()),dtype=np.uint8)
    frame = cv2.imdecode(image,-1)
    faces = classifier.detectMultiScale(frame,1.3,5)

    if faces is not None:    
        for x,y,w,h in faces:
            face = frame[y:y+h,x:x+w]
            cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,200),2)
            if len(data)<100:
                data.append(preprocess(face))
            else:
                cv2.putText(frame,'done',(100,100),
                            cv2.FONT_HERSHEY_PLAIN,4,
                            (255,255,255),5)
        
    cv2.imshow('video',frame)

    if cv2.waitKey(1)==ord('q'):
        break
cv2.destroyAllWindows()

data = np.array(data)

if data.shape[0]==100:
    
    name = input('enter the name of person: ')
    
    if not name=="flush":
        with open(name+'.p','wb') as f:
            pickle.dump(data,f)
        visualize_images(name)
