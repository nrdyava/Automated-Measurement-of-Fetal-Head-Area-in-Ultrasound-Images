# Automated measurement of fetal head area using Deep Learning
![segmented image](https://raw.githubusercontent.com/naveenrd/ultrasound-fetus-segmentation/master/segmented_image.JPG)

Semantic segmentation of Fetal head & Area determination using the [U-Net](https://arxiv.org/abs/1505.04597).
This model was trained using 800 training images. Random crop & random vertical flip methods are used for data augmentation.

The network was trained for 100 epochs using batch size=1 and obtained pixel accuracy of 96.95 % on the validation dataset which contains 200 images. The link to the trained weights is [here](https://drive.google.com/file/d/1-1EyBEFwcYASzuWcrETOv449BTrt0RaQ/view?usp=sharing).

The data is used from the [Grand Challenge](https://hc18.grand-challenge.org/).


