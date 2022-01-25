import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import os

from torchvision import transforms

path = "test/labels/"
def get_labels(path):
    files = os.listdir(path)
    for f in files:
        f_path = path + f
        img = cv2.imread(f_path)
        label = img.shape
        print(label)
        
    
        
        # print(type(img))

get_labels(path)
