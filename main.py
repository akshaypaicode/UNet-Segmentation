# -*- coding: utf-8 -*-
"""
Created on Mon May 14 23:42:28 2018

@author: akshay
"""

import numpy as np
import sys
import os
import tensorflow as tf
import keras
sys.path.insert(0, '/home/akshay/cnn_python/MRI_tools/')
sys.path.insert(0, '/home/akshay/cnn_python/models/')
import generateSlice
import labelHandle
import UNets
from keras.optimizers import Adam
from joblib import Parallel, delayed
from sklearn.utils import class_weight


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.85
set_session(tf.Session(config=config))

directoryImage = "/home/akshay/dataset/miccai/train/mri/"
directoryLabel = "/home/akshay/dataset/miccai/train/label/"

def parPool(numOrientations,numSlices,i,l,szSlice):
    parts = l[i].split('_')
    mriName = os.path.join(directoryImage,l[i])
    segName = os.path.join(directoryLabel,parts[0]+'_3_glm.mat')
    
    im = np.array([],dtype="float32")
    lab = np.array([],dtype="int32")
    
    for j in range(numSlices):
        
        imA, labA = generateSlice.getRandomOrientation(mriName,segName,numOrientations,szSlice)
        
        im = np.concatenate([im,imA]) if im.size else imA
        lab = np.concatenate([lab,labA]) if lab.size else labA
    
    im = np.reshape(im,(im.shape[0],im.shape[1],im.shape[2],1))
    lab = labelHandle.one_hot_encode_y(lab,135)
    
    return im,lab

#define parameters
szSlice = (336,336)
numSlices = 60
numOrientations = 1
l = os.listdir(directoryImage)

#create model
model = UNets.UNet(szSlice[0],szSlice[1],135)
model.compile(optimizer=Adam(lr=1e-4),loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()


for e in range(100):
    
    r = Parallel(n_jobs=5)(delayed(parPool)(numOrientations,numSlices,i,l,szSlice) for i in range(len(l)))
    trainMRI, trainLabel = zip(*r)

    print("Data collected and being prepared")
    
    train = np.array([],dtype="float16")
    target = np.array([],dtype="int16")
    
    for i in range(len(l)):
        train = np.concatenate([train,trainMRI[i]]) if train.size else trainMRI[i]
        target = np.concatenate([target,trainLabel[i]]) if target.size else trainLabel[i]

    print(train.shape)
    print(target.shape)
    c_w = class_weight.compute_class_weight('balanced',np.unique(np.argmax(train,3)), np.unique(np.argmax(train,3)))
    
    model.fit(train,target,epochs=1,batch_size=15,class_weight=c_w)    
    
    
    if not (e+1)%10:
        model.save_weights("UNetBasic"+str(e)+".h5")
    
    
