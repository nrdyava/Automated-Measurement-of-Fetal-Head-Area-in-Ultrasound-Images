from __future__ import print_function,division
import os
import sys
import random
from PIL import Image
import pandas as pd
from skimage import io,transform
import numpy as np
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
import matplotlib.pyplot as plt
from torchvision.transforms import functional as G
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import argparse

from Unet import Unet




def get_args():
    parser=argparse.ArgumentParser(description='Command line arguments for predicting Unet model perfomance')
    
    parser.add_argument('--visualize',type=bool,default=True,help='visualize predictions of the network ?',dest='visualize')
    parser.add_argument('--device',type=str,default='cpu',choices=['cpu','gpu'],help='device to run the model.',dest='device')
    parser.add_argument('--load',type=str,required=True,help='path of trained weights.',dest='load')
    parser.add_argument('--path',type=str,required=True,help='path of image to predict.',dest='path')
    parser.add_argument('--output_path',type=str,default=None,help='path to save the predicted image',dest='output_path')
    parser.add_argument('--pixel_size',type=float,default=None,help='pixel size of input image',dest='pixel_size')
    
    return parser.parse_args()

def area_calculator(sample):
    height,width=sample.shape
    count=0.0
    for i in range(height):
        for j in range(width):
            if sample[i][j]==255:
                count+=1
    return count 



if __name__=='__main__':
    args=get_args()
    
    img=io.imread(args.path)
    img_orig=np.array(img)
    height,width=img_orig.shape
    img=np.expand_dims(np.array(img),axis=-1)
    img=transform.resize(img,(572,572))
    img=img.transpose((2,0,1))
    img=torch.from_numpy(img).type(torch.FloatTensor)
    img=img.view(1,1,572,572)
    
    net=Unet()
    
    if args.device=='cpu':
        device='cpu'
        net.to(device)
        if args.load!=None:
            net.load_state_dict(torch.load(args.load,map_location=torch.device('cpu')))
        
    elif args.device=='gpu':
        device=torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        net.to(device)
        if args.load!=None:
            net.load_state_dict(torch.load(args.load,map_location=torch.device(device)))

    img=img.to(device)
    
    output=net(img)
    output=F.softmax(output,dim=1)
    output=torch.argmax(output,dim=1)
    output=output.view(388,388)
    output=output*255
    output=output.numpy()
    
    if args.pixel_size!=None:
        fetus_area=area_calculator(output)*(args.pixel_size**2)*(height/388)*(width/388)
    
    output=transform.resize(output,(height,width))
    

    if args.output_path!=None:
        io.imsave(args.output_path,output)


    if args.visualize:
        fig=plt.figure()
        if args.pixel_size!=None:
            text_output='fetus area: %f cm^2'%(round(fetus_area/100,4))
            fig.suptitle(text_output,fontsize=16,fontweight='bold')
        segmented=fig.add_subplot(1,2,1)
        segmented.set_title('segmented fetus')
        plot=plt.imshow(output,'gray')
        plt.axis('off')
        original=fig.add_subplot(1,2,2)
        original.set_title('original ultrasound image')
        plot=plt.imshow(img_orig,'gray')
        plt.axis('off')
        plt.show()
        




    
    
