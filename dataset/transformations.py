from __future__ import print_function,division
import os
import torch
import math
import sys
import random
from PIL import Image
import pandas as pd
from skimage import io,transform
import numpy as np
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt
from torchvision.transforms import functional as G



class Rescale(object):
    def __init__(self,output_size,process):
        assert isinstance(output_size,tuple)
        self.output_size=output_size
        self.process=process
        
    def __call__(self,sample):
        img,annotation=sample['img'],sample['annotation']
        
        new_h,new_w=self.output_size
        new_h,new_w=int(new_h),int(new_w)
        
        if self.process=='training':
            img=transform.resize(img,(new_h,new_w))
            annotation=transform.resize(annotation,(new_h,new_w))
            return {'img':img,'annotation':annotation}

        elif self.process=='validation':
            img=transform.resize(img,(new_h,new_w))
            annotation=transform.resize(annotation,(new_h-184,new_w-184))
            return {'img':img,'annotation':annotation}
            
            
        else:
            print('input the process parameter as training/validation.')
            
    

    
class RandomCrop(object):
    def __init__(self,output_size):
        assert isinstance(output_size,tuple)
        self.output_size=output_size
        
    def __call__(self,sample):
        img,annotation=sample['img'],sample['annotation']
        
        h,w=img.shape[:2]
        new_h,new_w=self.output_size
        
        top=np.random.randint(0,h-new_h)
        left=np.random.randint(0,w-new_w)
        
        img=img[top:top+new_h,left:left+new_w]
        annotation=annotation[top:top+new_h,left:left+new_w]
        annotation=transform.resize(annotation,(new_h-184,new_w-184))
           
        return {'img':img,'annotation':annotation} 
    
    
class Random_Vertical_Flip(object):
    def __init__(self,p=0.5):
        self.p=p
            
    def __call__(self,sample):
        img,annotation=sample['img'],sample['annotation']
        
        if random.random()<self.p:
            img=np.squeeze(img,axis=2)
            annotation=np.squeeze(annotation,axis=2)

            img=G.hflip(Image.fromarray(img))
            annotation=G.hflip(Image.fromarray(annotation))
            
            img=np.expand_dims(np.array(img),axis=-1)
            annotation=np.expand_dims(np.array(annotation),axis=-1)
            
            
            return {'img':img,'annotation':annotation}
        return sample
        

        
class ToTensor(object):
    def __call__(self,sample):
        img,annotation=sample['img'],sample['annotation']
            
        img = img.transpose((2, 0, 1))
        annotation=annotation.transpose((2,0,1))
        
        dtype = torch.FloatTensor

        return {'img':torch.from_numpy(img).type(dtype),'annotation':torch.from_numpy(annotation).type(dtype)}


