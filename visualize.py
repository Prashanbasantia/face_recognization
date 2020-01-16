# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 21:17:14 2020

@author: Prasan
"""


import numpy as np
import matplotlib.pyplot as plt
import cv2
import pickle
import random

def visualize_images(name):
    assert(type(name)==str),"name must be a string"
    with open(name+'.p','rb') as f:
        data = pickle.load(f)
    
    fig,axes = plt.subplots(nrows=10,ncols=5,figsize=(50,100))
    fig.tight_layout()
    for i in range(5):
        for j in range(10):
            axes[j][i].imshow(data[random.randint(0,data.shape[0]-1),:,:],cmap='gray')
            axes[j][i].axis('off')
        


visualize_images('asutosh')