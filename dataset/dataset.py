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
from torchvision.transforms import functional as F
from .transformations import *



class UltrasoundDataset(Dataset):
    def __init__(self,csv_file,root_dir,transform=None):
        self.file_names=pd.read_csv(csv_file)
        self.root_dir=root_dir
        self.transform=transform
        
    def __len__(self):
        return len(self.file_names)
    
    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx=idx.tolist()
         
        img_name=self.file_names.loc[idx,'filename']
        img_path=os.path.join(self.root_dir,img_name)
        
        root_dir=self.root_dir+'_annotations'
        annotation_path,_=img_name.split('.')
        annotation_path=annotation_path+'_Annotation.png'
        annotation_path=os.path.join(root_dir,annotation_path)
        
        img=io.imread(img_path)
        annotation=io.imread(annotation_path)
        annotation=(annotation==255)
        
        img=np.expand_dims(np.array(img),axis=-1)
        annotation=np.expand_dims(np.array(annotation),axis=-1)
        
        sample={'img':img,'annotation':annotation}
        
        if self.transform:
            sample=self.transform(sample)
        
        return sample

