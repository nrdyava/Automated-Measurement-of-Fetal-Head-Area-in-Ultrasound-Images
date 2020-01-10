import argparse
import os
import torch
import math
import sys
import random
import numpy as np
import torchvision
from torch.utils.data import Dataset,DataLoader
from torchvision import transforms,utils
from torchvision.transforms import functional as G
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from time import time

from Unet import Unet
from losses import cross_entropy,dice_loss
from dataset import UltrasoundDataset
from dataset.transformations import *




def get_args():
    parser=argparse.ArgumentParser(description='Command line arguments for training Unet')
    
    parser.add_argument('--lr',type=float,default=0.001,help='learning rate of the optimizer',dest='lr')
    parser.add_argument('--n_epochs',type=int,required=True,help='number of epochs',dest='n_epochs')
    parser.add_argument('--device',type=str,default='cpu',choices=['cpu','gpu'],help='device to train the model',dest='device')
    parser.add_argument('--load',type=str,default=None,help='path of pre-trained weights',dest='load')
    parser.add_argument('--bs',type=int,default=1,help='batch size of dataloader',dest='bs')
    parser.add_argument('--save',type=str,required=True,help='path to save trained weights',dest='save')
    
    return parser.parse_args()


def train_n_epochs(n_epochs):
    print('training Network:\n')
    for epoch in range(n_epochs):
        init=time()
        training_loss=0.0
        validation_loss=0.0
        mean_validation_accuracy=0.0
        mean_training_accuracy=0.0
        
        for i,batch_data in enumerate(training_dataloader):
            img,target=batch_data['img'].to(device),((batch_data['annotation'].view(batch_data['img'].shape[0],388,388)).type(torch.LongTensor)).to(device)

            optimizer.zero_grad()
        
            output=net(img)
            loss=criterion(output,target)
            loss.backward()
            optimizer.step()
            
            training_loss+=loss.item()
            
            output=torch.argmax(output,dim=1)
            output=(output==target)
            correct=int(torch.sum(output))
            total=target.shape[0]*150544
            mean_training_accuracy+=correct/total
        
        
        for j,batch_data in enumerate(validation_dataloader):
            img,target=batch_data['img'].to(device),((batch_data['annotation'].view(batch_data['img'].shape[0],388,388)).type(torch.LongTensor)).to(device)

            optimizer.zero_grad()
        
            output=net(img)
            loss=criterion(output,target)
            
            validation_loss+=loss.item()
            
            output=F.softmax(output,dim=1)
            output=torch.argmax(output,dim=1)
            output=(output==target)
            correct=int(torch.sum(output))
            total=target.shape[0]*150544
            mean_validation_accuracy+=correct/total
        
        end=time()
        duration=(end-init)/60
        mean_training_accuracy=(mean_training_accuracy/(i+1))*100
        mean_validation_accuracy=(mean_validation_accuracy/(j+1))*100
        print('\n# epoch: %d  training loss:%f & validation loss:%f & training_accuracy:%f & validation_accuracy:%f duration:%f'%(epoch+1,training_loss,validation_loss,mean_training_accuracy,mean_validation_accuracy,duration))

                  
    print('\ntraining for %d epochs is done.'%(n_epochs))

    
    
if __name__=='__main__':
    args=get_args()
    criterion=cross_entropy()
    net=Unet()
    optimizer=optim.Adam(net.parameters(),lr=args.lr)
    
    
    
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
        
        
        
    
    usi_training_dataset=UltrasoundDataset(csv_file='data/training_set_pixel_size.csv',
                                     root_dir='data/training_set',
                                     transform=transforms.Compose([
                                         Rescale((800,600),process='training'),
                                         RandomCrop((572,572)),
                                         Random_Vertical_Flip(),
                                         ToTensor()
                                     ]))
    
    
    usi_validation_dataset=UltrasoundDataset(csv_file='data/validation_set_pixel_size.csv',
                                     root_dir='data/validation_set',
                                     transform=transforms.Compose([
                                         Rescale((572,572),process='validation'),
                                         ToTensor()
                                     ]))
    
    
    training_dataloader=DataLoader(usi_training_dataset,batch_size=args.bs,shuffle=False,num_workers=4)
    
    validation_dataloader=DataLoader(usi_validation_dataset,batch_size=args.bs,shuffle=False,num_workers=4)
    
    train_n_epochs(args.n_epochs)
    torch.save(net.state_dict(),args.save)
