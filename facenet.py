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

_FLOATX = 'float32'

# Загрузка Facenet-модели и весов
model=load_model(os.path.join(r'keras\model',r'facenet_keras.h5'))
model.load_weights(os.path.join(r'keras\weights',r'facenet_keras_weights.h5'))

model.summary()

base_dir=r'dataset'

def Standardize(inp):
    if inp.ndim==3:
        axis=(0,1,2)
    elif inp.ndim==4:
        axis=(1,2,3)
    mean=np.mean(inp,axis=axis,keepdims=True)
    std=np.std(inp,axis=axis,keepdims=True)
    inp=(inp-mean)/std
    return inp

def L2_Norm(inp):
    for i in range(len(inp)):
        inp[i]=inp[i]/np.sqrt(np.sum(np.square(inp[i])))
    
    return inp

def Euclied_dist(inp1,inp2):
    return np.sqrt(np.sum(np.square(inp1-inp2)))

# Определение детектора и предиктора
detector=dlib.get_frontal_face_detector()
shape_predictor=dlib.shape_predictor(os.path.join(r'dlib',r'shape_predictor_68_face_landmarks.dat'))
face_aligner=FaceAligner(shape_predictor)

# Построение датасета
if os.path.exists('data\FImages.npy') == False:
    images=[]
    for i in os.listdir(base_dir):
        for j in os.listdir(os.path.join(base_dir,i)):
            img=cv2.imread(os.path.join(base_dir,i,j))
            print(i,j)
            faces=detector(img,1)
            f=faces[0]
            print(len(faces))
            x,y,w,h=f.left(),f.top(),(f.right()-f.left()),(f.bottom()-f.top())
            temp=face_aligner.align(img,cv2.cvtColor(img,cv2.COLOR_BGR2GRAY),f)
            temp=cv2.resize(cv2.cvtColor(temp,cv2.COLOR_BGR2GRAY),(160,160))
            cv2.imwrite(os.path.join(base_dir,r'temp.jpg'),temp)
            temp=cv2.imread(os.path.join(base_dir,r'temp.jpg'))
            os.remove(os.path.join(base_dir,r'temp.jpg'))
            images.append(temp)

    images=Standardize(np.array(images))

    np.save(os.path.join(r'data',r'FImages'),images)

    images=np.load(os.path.join(r'data',r'FImages.npy'))
else:
    images=np.load(os.path.join(r'data',r'FImages.npy'))

# Преобразование лейблов для всех лиц
labels=[]
for i in os.listdir(base_dir):
    labels.extend([i]*len(os.listdir(os.path.join(base_dir,i))))
    
le=LabelEncoder().fit(labels)    
y=le.transform(labels)

# Генерация вставок для каждого из обучаемых лиц 
embs=[]
for i in range(len(images)):
    t=L2_Norm(np.array(model.predict(images[i:i+1])))
    embs.append(t)
embs=np.reshape(np.array(embs),(len(embs),128))   

# Рассчет средних значений каждого лица
labels_emb={}
i=0
c=0
for j in os.listdir(base_dir):
    l=len(os.listdir(os.path.join(base_dir,j)))
    print(l)
    # Принять значение всех вложений конкретного лица
    labels_emb[c]=np.mean(embs[i:i+l],axis=0)
    c+=1
    i+=l

# Определение SVC модели для классификации и обучения с помещенными данными вставок
Smodel=SVC(kernel='linear',probability=True,decision_function_shape='ovo')
clf=Smodel.fit(embs,y)

# Рассчет порогового значения расстояния
for i in range(len(embs)):
    for j in range(len(labels_emb)):
        print("{} and {} :{}".format(i,j,Euclied_dist(embs[i],labels_emb[j])))
