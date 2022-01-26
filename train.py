import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import os
from utils import get_labels

from torchvision import transforms

def train():
    get_labels("./test/")
    pass


if __name__ == "__main__":
    train()