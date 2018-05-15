# -*- coding: utf-8 -*-
"""
Created on Sun May 13 00:36:58 2018

@author: akshay
"""

import sys
sys.path.insert(0, '/home/akshay/cnn_python/MRI_tools/')
sys.path.insert(0, '/home/akshay/cnn_python/evaluation/')
sys.path.insert(0, '/home/akshay/cnn_python/models/')
import loader
import os
import UNets
import numpy as np
import keras
import tensorflow as tf
import generateSlice
from keras.optimizers import Adam
from scipy.io import savemat
from PIL import Image


import tensorflow as tf
from keras.backend.tensorflow_backend import set_session
config = tf.ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.2
set_session(tf.Session(config=config))

#get the test image
directoryImage = "/home/akshay/dataset/miccai/test/mri/"
directoryLabel = "/home/akshay/dataset/miccai/test/label/"
l = os.listdir(directoryImage)


szSlice = (336,336,336)

model = UNets.UNet(szSlice[0],szSlice[1],135)
model.summary()
model.compile(loss='categorical_crossentropy', optimizer=Adam(lr=1e-4, beta_1=0.9),metrics=['categorical_accuracy'])
print("Model created")

model.load_weights("UNetBasic49.h5")

for j in range(1):
    parts = l[j].split('_')
    mriName = os.path.join(directoryImage,l[j])
    segName = os.path.join(directoryLabel,parts[0]+'_3_glm.mat')
    
    header, mri = loader.load_nii(mriName)
    header, label = loader.load_mat(segName)

    mm = np.mean(mri)
    ss = np.std(mri)
    mri = (mri-mm)/ss

    
    pred_true = np.zeros(szSlice, dtype='int16')
    pred_true[0:label.shape[0],0:label.shape[1],0:label.shape[2]] = label
    #label = []
    #directions = np.random.uniform(-45,45,size=(3-1,3))
    #directions = np.vstack(((0,0,0),directions))
    directions = [(0,0,0)]
    print(len(directions))
    pred = np.zeros((len(directions)*3,szSlice[0],szSlice[1],szSlice[2]), dtype='int16')
    pp_true = np.zeros(szSlice, dtype='int16')


    count = 0
    for k in range(1):
        mri_for_pred = np.zeros(szSlice)
        for m in range(len(directions)):
            for l in range(mri.shape[k]):
                if k==0:
                    slice_rot_p = generateSlice.generateSlice(mri,[directions[m]],(1,1,1),0,(szSlice[0],szSlice[1]),seg=label,dd=k,sliceNum=l)
                    print(pp_true.shape)
                    slice_rot = np.reshape(slice_rot_p[0],(1,szSlice[0],szSlice[1],1))
                    pp_true[l,:,:] = (slice_rot_p[1])
                    slice_rot = model.predict(slice_rot.astype("float16"),batch_size=1,verbose=0)
                    mri_for_pred[l,:,:] = np.argmax(np.squeeze(slice_rot),2)
                    print(np.max(mri_for_pred[l,:,:]))
                if k==1:
                    slice_rot = generateSlice.generateSlice(mri,directions[m],(1,1,1),0,(szSlice[0],szSlice[1]),seg=None,dd=k,sliceNum=l)
                    slice_rot =	np.reshape(slice_rot[0],(1,szSlice[0],szSlice[1],1))
                    slice_rot = model.predict(slice_rot,batch_size=1,verbose=0)
                    mri_for_pred[:,l,:] = np.argmax(np.squeeze(slice_rot),2)
                if k==2:
                    slice_rot = generateSlice.generateSlice(mri,directions[m],(1,1,1),0,(szSlice[0],szSlice[1]),seg=None,dd=k,sliceNum=l)
                    slice_rot =	np.reshape(slice_rot[0],(1,szSlice[0],szSlice[1],1))
                    slice_rot = model.predict(slice_rot,batch_size=1,verbose=0)
                    mri_for_pred[:,:,l] = np.argmax(np.squeeze(slice_rot),2)
                
                print("predicted volume..." + str(m) + "in direction.." + str(k))

        pred[count,:,:,:] = mri_for_pred.astype('int32')
        count = count+1

    #int("computing the dice")
    #dice = metrics.dice_all_parallelized(pred_true,np.squeeze(mri_for_pred), smooth=1.0, ignore_zero=True)
    #print(dice)
    im = Image.fromarray(np.squeeze(mri_for_pred[128,:,:]))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("pred.png")
    im = Image.fromarray(np.squeeze(pp_true[128,:,:]))
    if im.mode != 'RGB':
        im = im.convert('RGB')
    im.save("true.png")


