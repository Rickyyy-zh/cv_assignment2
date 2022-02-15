from math import fabs
# from msilib.schema import Class
from smtpd import DebuggingServer
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

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    #BasicBlock and BottleNeck block
    #have different output size
    #we use class attribute expansion
    #to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BottleNeck(nn.Module):
    """Residual block for resnet over 50 layers
    """
    expansion = 4
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, stride=stride, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BottleNeck.expansion, kernel_size=1, bias=False),
            nn.BatchNorm2d(out_channels * BottleNeck.expansion),
        )

        self.shortcut = nn.Sequential()

        if stride != 1 or in_channels != out_channels * BottleNeck.expansion:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BottleNeck.expansion, stride=stride, kernel_size=1, bias=False),
                nn.BatchNorm2d(out_channels * BottleNeck.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))
    
class DeBasicBlock(nn.Module):
    
    expansion = 1
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.deresidual_function = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size = 3, stride = stride, padding=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_channels, out_channels * DeBasicBlock.expansion, kernel_size = 3, stride = 1, padding=2, bias=False),
            nn.BatchNorm2d(out_channels*DeBasicBlock.expansion)
        )
        
        #shortcut
        self.shortcut = nn.Sequential()

        #the shortcut output dimension is not the same with residual function
        #use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.ConvTranspose2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.deresidual_function(x) + self.shortcut(x))

        
        
    
class Res_segNet(nn.Module):
    
    def __init__(self, block, deblock, num_block):
        super().__init__()
        self.in_channels = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True))
        
        self.conv2_x = self._make_layer(block, 64, num_block[0], 1)
        self.conv3_x = self._make_layer(block, 128, num_block[1], 2)
        self.conv4_x = self._make_layer(block, 256, num_block[2], 2)
        self.conv5_x = self._make_layer(block, 512, num_block[3], 2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        # self.fc = nn.Linear(512 * block.expansion, num_classes)
        self.conv_mid = nn.Sequential(nn.Conv2d(512,512, kernel_size=1,bias=False),
            nn.ReLU(inplace=True)
            )
        self.conv6_x = nn.Sequential(
            nn.ConvTranspose2d(512, 512, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(512, 256, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 256, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(256),
            nn.ConvTranspose2d(256, 256, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv7_x = nn.Sequential(
            nn.ConvTranspose2d(256, 256, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(256, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(128),
            nn.ConvTranspose2d(128, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True)
        )
        self.conv8_x = nn.Sequential(
            nn.ConvTranspose2d(128, 128, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.ConvTranspose2d(128, 64, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 64, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=3,stride=1,padding=0),
            nn.ReLU(inplace=True)
        )
        self.pixel_class_x = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 3, kernel_size=3, padding=0, bias=False),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True),
        )
        self.softmax = nn.Sequential(nn.Softmax(1))
    def _make_layer(self, block, out_channels, num_blocks, stride):
        """make resnet layers(by layer i didnt mean this 'layer' was the
        same as a neuron netowork layer, ex. conv layer), one layer may
        contain more than one residual block
        Args:
            block: block type, basic block or bottle neck block
            out_channels: output depth channel number of this layer
            num_blocks: how many blocks per layer
            stride: the stride of the first block of this layer
        Return:
            return a resnet layer
        """

        # we have num_block blocks per layer, the first block
        # could be 1 or 2, other blocks would always be 1
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.expansion

        return nn.Sequential(*layers)
    
    def forward(self, x):
        output = self.conv1(x)
        output = self.conv2_x(output)
        output = self.conv3_x(output)
        output = self.conv4_x(output)
        output = self.conv5_x(output)
        output = self.avg_pool(output)
        # output = output.view(output.size(0), -1)
        # output = self.fc(output)
        output = self.conv_mid(output)
        output = self.conv6_x(output)
        output = self.conv7_x(output)
        output = self.conv8_x(output)
        output = self.pixel_class_x(output)
        output = self.softmax(output)
        return output

def Res18_segNet():
    
    return Res_segNet(BasicBlock, DeBasicBlock, [2,2,2,2])
