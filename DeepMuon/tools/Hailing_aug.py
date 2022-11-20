'''
Author: airscker
Date: 2022-10-14 20:47:38
LastEditors: airscker
LastEditTime: 2022-10-14 20:48:45
Description: NULL

Copyright (c) 2022 by airscker, All Rights Reserved. 
'''
import numpy as np

def Rotate90(image,label):
    """
    Rotate the image 90 degrees clockwise and return the rotated image and label.
    @param image - the image to rotate.
    @param label - the label to rotate.
    @returns the rotated image and label.
    """
    image=np.transpose(np.array(image),(1,0,2,3))
    image=image[::-1,...]
    label=np.array([-label[1],label[0],label[2]])
    return image,label

def Rotate180(image,label):
    """
    Rotate the image 180 degrees. Also rotate the label 180 degrees.
    @param image - the image to rotate.
    @param label - the label to rotate.
    @returns the rotated image and label.
    """
    image=np.array(image)
    image=image[::-1,::-1,:,:]
    label=np.array([-label[0],-label[1],label[2]])
    return image,label

def Flip(image,label):
    """
    Flip the image and label.
    @param image - the image to flip
    @param label - the label to flip
    @returns the flipped image and label
    """
    image=np.array(image)
    image=image[:,:,::-1,:]
    label=np.array([label[0],label[1],-label[2]])
    return image,label
