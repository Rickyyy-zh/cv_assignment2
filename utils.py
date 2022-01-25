import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import os

from torchvision import transforms
path = "train_data/images/"
def get_labels(path):
    files = os.listdir(path)
    for f in files:
        f_path = path + f
        img = cv2.imread(f_path)
        
        print(img)

print(cv2.__version__)

get_labels(path)
