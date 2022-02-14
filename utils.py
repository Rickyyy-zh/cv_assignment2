from cv2 import imshow
from matplotlib.pyplot import axis
import numpy as np
import cv2
# from sklearn.metrics import label_ranking_average_precision_score
import torch
import torchvision
import torch.nn as nn
import io
import os

from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

path = "test/labels/1.png"

def get_labels(path):

    img = cv2.imread(path)
    label = np.zeros(img.shape)
    label[np.all(img == np.array((0,0,0)),axis=-1)] = np.array((1,0,0))   # backgroud
    label[np.all(img == np.array((0,0,128)),axis=-1)] = np.array((0,1,0))     # coal
    label[np.all(img == np.array((0,128,0)),axis=-1)] = np.array((0,0,1))     # gangue
    
    return label
        
class Coal_dataset(Dataset):

    def __init__(self, root_dir, transform = None):
        self.root_dir = root_dir   
        self.transform = transform 
        self.images = os.listdir(self.root_dir)
    
    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        img = cv2.imread(img_path)
        label = get_labels(img_path.replace("images", "labels").replace("jpg", "png"))
        sample = {'image':img,'label':label}
        
        if self.transform:
            sample["image"] = self.transform(sample["image"])#对样本进行变换
        return sample #返回该样本

test = get_labels(path)
