from cv2 import imshow
import numpy as np
import cv2
from sklearn.metrics import label_ranking_average_precision_score
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
        label = cv2.imread(f_path)
        label[label == np.array((0,0,0))] = 0   # backgroud
        label[label == np.array((0,0,128))] = 1     # coal
        label[label == np.array((0,128,0))] = 2     # gangue

        print(label)
        cv2.imshow("test",label)
        cv2.waitKey(0)
        
        # print(type(img))

get_labels(path)
