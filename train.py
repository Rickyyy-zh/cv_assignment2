import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from utils import Coal_dataset, get_labels
from model import Res34net

from torchvision import transforms

def train():
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")
    print("The device is ", device)

    data_transform = {
        "train": transforms.Compose([transforms.RandomResizedCrop(224),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])]),

        "val": transforms.Compose([ transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    
    images_path = "./test/"
    train_dataset = Coal_dataset(root_dir= images_path, transform= data_transform["train"])
    train_data_num = len(train_dataset)
    print(train_data_num)

if __name__ == "__main__":
    train()