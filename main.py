# coding: utf-8

import tensorflow as tf
import keras
from keras import applications
import os
from keras.models import load_model,model_from_json
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import inline
import cv2
import dlib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
import imutils
from imutils.face_utils import FaceAligner
import time

from telegrambot import*
from facenet import*

alarm_time=0
# Распознавание нескольких лиц в реальном времени с использованием веб-камеры
cam=cv2.VideoCapture(0)
while True:
    ret,img=cam.read()
    temp1=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces=detector(img,1)
    x,y,w,h,F=[],[],[],[],[]
    Fimages=[]
    flag=False
    for i,f in enumerate(faces):
        flag=True
        a,b=f.left(),f.top()
        x.append(a)
        y.append(b)
        w.append(f.right()-a)
        h.append(f.bottom()-b)
        F.append(f)

    if flag==True:
        for i in range(len(x)):
            temp=face_aligner.align(temp1,temp1,F[i])
            temp=cv2.resize(temp,(160,160))
            cv2.imwrite(os.path.join(r'temp',r'temp.jpg'),temp)
            temp=cv2.imread(os.path.join(r'temp',r'temp.jpg'))
            os.remove(os.path.join(r'temp',r'temp.jpg'))
            Fimages.append(temp)
            
        Fimages=Standardize(np.array(Fimages))
        facenetpred=L2_Norm(model.predict(Fimages))
        pred=[]
        dist=[]
        for i in range(len(facenetpred)):
            t=clf.predict(np.reshape(facenetpred[i],(1,len(facenetpred[i]))))
            #print(t)
            tt=clf.predict_proba(np.reshape(facenetpred[i],(1,len(facenetpred[i]))))
            # FMax and SMax первая и вторая высокая возможность прогнозирования
            FMax=np.max(tt)
            #print(FMax)
            SMax=np.max(np.delete(tt,tt.argmax()))
            # Рассчет значений незнакомого лица
            if Euclied_dist(labels_emb[t[0]],facenetpred[i])<0.6 and FMax-SMax>0.3:  
                pred.append(le.inverse_transform(t)[0])
            else:
                pred.append('Unknown')
            dist.append(str(FMax))
                
        c=0
        n='Unknown'
        for i in pred:
            if n in pred[c]:
                cv2.putText(img,i,(x[c]-10,y[c]-10),cv2.FONT_HERSHEY_PLAIN,2,(125,176,209),2)
                cv2.rectangle(img,(x[c],y[c]),(x[c]+w[c],y[c]+h[c]),(255,0,255),2)
                time2=time.time()
                if (time2-alarm_time)>10:
                    #print('Обнаружено неустановленное лицо')
                    alarm_time=time.time()
                    #send_screen(img)
            else:
                cv2.putText(img,i,(x[c]-10,y[c]-40),cv2.FONT_HERSHEY_PLAIN,2,(125,176,209),2)
                cv2.putText(img,'dist: ' + "{0:.2f}".format(float(dist[c])),(x[c]-10,y[c]-10),cv2.FONT_HERSHEY_PLAIN,2,(79,79,196),2)
                cv2.rectangle(img,(x[c],y[c]),(x[c]+w[c],y[c]+h[c]),(255,0,255),2)

            c+=1
            
    cv2.imshow('Video',img)
    if cv2.waitKey(10) == 27:
        break
    
cv2.destroyAllWindows() 
cam.release()
