import zipfile
import wget
import os
import numpy as np
import pandas as pd
import shutil
from skimage import io

class download_data(object):
    def __call__(self,url=None,folder=None,root_dir=os.getcwd()):
        
        working_directory=root_dir
        
        if os.path.exists(os.path.join(root_dir,'data')):
            root_dir=os.path.join(root_dir,'data')
        else:
            os.mkdir('data')
            root_dir=os.path.join(root_dir,'data')
        
        os.chdir(root_dir)
        
        if folder!=None:
            os.mkdir(folder)
            os.chdir(os.path.join(root_dir,folder))
            downloaded=wget.download(url)
        else:
            downloaded=wget.download(url)
        
        
        if zipfile.is_zipfile(downloaded):
            with zipfile.ZipFile(downloaded,'r') as zipped:
                zipped.extractall()
            os.remove(downloaded)
        
        os.chdir(working_directory)


class split_annotations(object):
    def __call__(self,csv_file=None):
        working_dir=os.getcwd()
        
        file_names=pd.read_csv(csv_file)
        
        os.mkdir('data/training_set_annotations')
        os.chdir('data/training_set')
        
        for i in range(len(file_names)):
            img_name,_=file_names.iloc[i,0].split('.')
            img_name=img_name+"_Annotation.png"
            shutil.move(img_name,os.path.join(working_dir,'data/training_set_annotations'))
        
        os.chdir(working_dir)

def annotator(img):
    height,width=img.shape
    for i in range(height):
        indices=index_finder(img[i],width)
        if indices:
            img[i][indices['left_index']:indices['right_index']]=255
    
    return img


def index_finder(row,width):
    for i in range(width):
        if row[i]==255:
            left_index=i
            for j in range(width):
                if row[width-1-j]==255:
                    right_index=width-1-j
                    return {'left_index':left_index,'right_index':right_index}



class mask_annotations(object):
    def __call__(self,csv_file=None):
        working_dir=os.getcwd()
        
        file_names=pd.read_csv(csv_file)
        os.chdir('data/training_set_annotations')
        
        for i in range(len(file_names)):
            img_name,_=file_names.iloc[i,0].split('.')
            img_name=img_name+"_Annotation.png"
            
            img=io.imread(img_name)
            
            img=annotator(img)
            os.remove(img_name)
            io.imsave(img_name,img)
            
        os.chdir(working_dir)


class data_splitter(object):
    def __call__(self,csv_file=None,valid_frac=None):
       
        working_dir=os.getcwd()
        
        file_names=pd.read_csv(csv_file)
        valid_file_names=file_names.sample(frac=valid_frac,random_state=1)
        train_file_names=file_names.drop(valid_file_names.index)
        
        valid_file_names.to_csv('data/validation_set_pixel_size.csv')
        train_file_names.to_csv('data/training_set_pixel_size.csv')
        
        os.mkdir('data/validation_set')
        os.mkdir('data/validation_set_annotations')
        os.chdir('data/training_set')
        
        for i in range(len(valid_file_names)):
            shutil.move(valid_file_names.iloc[i,0],os.path.join(working_dir,'data','validation_set'))
        
        os.chdir(working_dir)
        os.chdir('data/training_set_annotations')
    
        for i in range(len(valid_file_names)):
            img_name,_=valid_file_names.iloc[i,0].split('.')
            img_name=img_name+"_Annotation.png"
            shutil.move(img_name,os.path.join(working_dir,'data','validation_set_annotations'))
        
        os.chdir(working_dir)



url_test_set='https://zenodo.org/record/1327317/files/test_set.zip?download=1'
url_test_set_pixel_size='https://zenodo.org/record/1327317/files/test_set_pixel_size.csv?download=1'
url_training_set='https://zenodo.org/record/1327317/files/training_set.zip?download=1'
url_training_set_pixel_size_and_HC='https://zenodo.org/record/1327317/files/training_set_pixel_size_and_HC.csv?download=1'


downloader=download_data()
annotations_splitter=split_annotations()

print('\nDownloading data:')
downloader(url_test_set)
downloader(url_test_set_pixel_size)
downloader(url_training_set)
downloader(url_training_set_pixel_size_and_HC)


annotations_splitter(csv_file='data/training_set_pixel_size_and_HC.csv')

print('\nMasking annotations:')
masker=mask_annotations()
masker(csv_file='data/training_set_pixel_size_and_HC.csv')


data_splitter=data_splitter()
data_splitter(csv_file='data/training_set_pixel_size_and_HC.csv',valid_frac=0.2)

print('\n')


