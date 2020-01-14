# Automated measurement of fetal head area using Deep Learning
![segmented image](https://raw.githubusercontent.com/naveenrd/ultrasound-fetus-segmentation/master/segmented_image.JPG)

Semantic segmentation of Fetal head & Area determination using the [U-Net](https://arxiv.org/abs/1505.04597).
This model was trained from scratch using 800 training images. Random crop & random vertical flip methods are used for data augmentation.

The network was trained for 100 epochs using batch size=1 and obtained pixel accuracy of 96.95 % on the validation dataset which contains 200 images. The link to the trained weights is [here](https://drive.google.com/file/d/1-1EyBEFwcYASzuWcrETOv449BTrt0RaQ/view?usp=sharing).

The data is used from the [Grand Challenge](https://hc18.grand-challenge.org/).

##Usage
**Programming Language used: Python3**
###Training

'''shell script
>python3 trainer.py -h
usage: trainer.py [-h] [--lr LR] --n_epochs N_EPOCHS [--device {cpu,gpu}]
                  [--load LOAD] [--bs BS] --save SAVE

Command line arguments for training Unet

optional arguments:
  -h, --help           show this help message and exit
  --lr LR              learning rate of the optimizer
                              (default learning rate=0.001)
  --n_epochs N_EPOCHS  number of epochs
                              (required: True)
  --device {cpu,gpu}   device to train the model
                              (default device:'cpu')
  --load LOAD          path of pre-trained weights
                              (default= None,it will start from scratch)
  --bs BS              batch size of dataloader
                              (default= 1)
  --save SAVE          path to save trained weights
                              (required: True)
'''
One can specify which model to use with '--load WEIGHTS.pth'


