from math import fabs
from cv2 import FlannBasedMatcher
import torch
import torchvision
import torch.nn as nn
import torch.utils.model_zoo as model_zoo

def Res34net(pretrain=False, **kwargs):
    model = torchvision.models.resnet34(pretrained=pretrain)
    if pretrain:
        model = torchvision.models.resnet34(pretrained=pretrain)
    
    return model

