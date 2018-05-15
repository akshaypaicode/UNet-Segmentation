# -*- coding: utf-8 -*-
"""
Created on Wed May  9 22:44:45 2018

@author: akshay
"""

import sys
import loader
import numpy as np
import math
import generateSlice
from scipy.interpolate import RegularGridInterpolator


def interpolate3D(x,y,z,volume,voxSize,method,size):
    xi = np.arange(0,volume.shape[0],voxSize[0])
    yi = np.arange(0,volume.shape[1],voxSize[1])
    zi = np.arange(0,volume.shape[2],voxSize[2])
    im_intrp = RegularGridInterpolator((xi, yi, zi), volume,bounds_error=False,fill_value=0,method=method)    
    imNew = im_intrp((x,y,z))
    imNew = np.reshape(imNew,size)
    return imNew
        

def rotationMatrices(theta):
    
    theta = np.deg2rad(theta)
    rotX = [[math.cos(theta[0]),-math.sin(theta[0]),0], [math.sin(theta[0]),math.cos(theta[0]),0], [0,0,1]]
    rotY = [[math.cos(theta[1]),0,math.sin(theta[1])], [0,1,0], [-math.sin(theta[1]),0,math.cos(theta[1])]]
    rotZ = [[1,0,0], [0,math.cos(theta[2]),-math.sin(theta[2])], [0,math.sin(theta[2]),math.cos(theta[2])]]
    rot = np.dot(rotX,rotY).dot(rotZ)

    return rot


    
def generateSlice(volume,directions,voxSize,scale,szSlice,seg=None,dd=None,sliceNum=None,limit=None):

    if scale==1:
        mm = np.mean(volume)
        ss = np.std(volume)
        volume = (volume-mm)/ss
    if dd is not None:
        d = dd
    else:
        d = np.random.randint(3,size=1)    
    if sliceNum is not None:
        s = sliceNum
    else:
        if limit is not None:
            s = np.random.randint(limit[int(d)][0],limit[int(d)][1],size=1)
        else:
            s = np.random.randint(volume.shape[int(d)],size=1)

    #create a matrix where the views are stored.
    imArray = np.zeros([len(directions),szSlice[0], szSlice[1]])
    labArray = np.zeros([len(directions),szSlice[0], szSlice[1]])
    for i in range(len(directions)):        
        if d==0:
            slice = np.squeeze(volume[s,:,:])
            z = np.arange(0, volume.shape[2], voxSize[2])
            y = np.arange(0, volume.shape[1], voxSize[1])
            x = s
        elif d==1:
            slice = np.squeeze(volume[:,s,:])
            x = np.arange(0, volume.shape[0], voxSize[0])
            z = np.arange(0, volume.shape[2], voxSize[2])
            y = s
        elif d==2:
            slice = np.squeeze(volume[:,:,s])
            x = np.arange(0, volume.shape[0], voxSize[0])
            y = np.arange(0, volume.shape[1], voxSize[1])
            z = s        
        #center

        c = tuple(ti/2 for ti in volume.shape)
        x = x-c[0]
        y = y-c[1]
        z = z-c[2]
        
        #create a meshgrid
        xx, yy, zz = np.meshgrid(x,y,z)
        xyz = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()])
        rot = rotationMatrices(directions[i])
        xy_t = np.dot(rot, xyz)
        
        im0 = interpolate3D(xy_t[0]+c[0], xy_t[1]+c[1],xy_t[2]+c[2], volume, voxSize,"linear",(slice.shape[0],slice.shape[1]))
        imArray[i,0:slice.shape[0],0:slice.shape[1]] = np.reshape(im0,(slice.shape[0],slice.shape[1]))
        if seg is not None:
            im1 = interpolate3D(xy_t[0]+c[0], xy_t[1]+c[1],xy_t[2]+c[2], seg, voxSize,"nearest",(slice.shape[0],slice.shape[1]))
            labArray[i,0:slice.shape[0],0:slice.shape[1]] = np.reshape(im1,(slice.shape[0],slice.shape[1]))
            return imArray, labArray
        else:
            return imArray,None
        
        

def getRandomOrientation(mriName,segName,numOrientations,szSlice):

    #reinterpolate to isotropic grid                                                                                                       
    #z = np.arange(0, 256, 1)
    #y = np.arange(0, 256, 1)
    #x = np.arange(0, 256, 1)
    #xx,yy,zz = np.meshgrid(x,y,z)
    header, mriIsotropic = loader.load_nii(mriName)
    header, labelIsotropic = loader.load_mat(segName)

    
    x,y,z = np.where(labelIsotropic>0)

    limit = [[np.min(x),np.max(x)],[np.min(y),np.max(y)],[np.min(z),np.max(z)]]
    #mriIsotropic = interpolate3D(xx.ravel(),yy.ravel(),zz.ravel(),mriIsotropic,(1,1,1),"linear",(256,256,256))
    #labelIsotropic = interpolate3D(xx.ravel(),yy.ravel(),zz.ravel(),labelIsotropic,(1,1,1),"nearest",(256,256,256))

    directions = np.random.uniform(-45,45,size=(numOrientations-1,3))
    directions = np.vstack(((0,0,0),directions))
    voxSize = (1,1,1)
    imArray, labArray = generateSlice(mriIsotropic,directions,voxSize,1,szSlice,seg=labelIsotropic,dd=None,sliceNum=None,limit=limit)
    
    return imArray, labArray

    

    

            
        
        
        
        
        
    
