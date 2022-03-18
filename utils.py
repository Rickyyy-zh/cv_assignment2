from matplotlib.pyplot import axis
import numpy as np
import cv2
# from sklearn.metrics import label_ranking_average_precision_score
import torch
from torchvision.transforms import InterpolationMode
import torch.nn as nn
import io
import os
import random
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import torch.cuda.amp as amp

path = "test/images/"

def get_labels(path):

    img = cv2.imread(path)
    
    img = img[:,:,::-1].copy()
    # img = cv2.resize(img, (320,320))
    label = np.zeros(img.shape)
    label[np.all(img == np.array((0,0,0)),axis=-1)] = np.array((1,0,0))   # backgroud
    label[np.all(img == np.array((128,0,0)),axis=-1)] = np.array((0,1,0))     # coal
    label[np.all(img == np.array((0,128,0)),axis=-1)] = np.array((0,0,1))     # gangue
    # label = label.transpose(2, 0, 1)
    
    label = Image.fromarray(np.uint8(label))

    return label

def addPRendpoint(p_list, r_list):
    p_list.append(p_list[-1])
    p_list.insert(0,0.0)
    p_list.append(1.0)
    p_list.insert(0,0.0)
    r_list.append(0.0)
    r_list.insert(0,r_list[0]) 
    r_list.append(0.0)
    r_list.insert(0,1.0) 
    return p_list, r_list

def cal_pr(res, label, threshold = 0.5):
    precision = np.zeros((1,3), dtype=np.float32)
    recall = np.zeros((1,3), dtype=np.float32)
    fpr = np.zeros((1,3), dtype=np.float32)
    for cls in range(3):
        label_cls = label[:,cls,:,:]
        res_cls = res[:, cls, :,:]
        res_cls= res_cls >= threshold
        # res_cls = np.zeros(label.shape).sum(axis=1)
        # res_cls[np.where(res == cls)] = 1
        _tp = np.sum(res_cls[label_cls == 1.] == True)
        _fn = np.sum(res_cls[label_cls == 1.] == False)
        _fp = np.sum(res_cls[label_cls == 0.] == True)
        _tn = np.sum(res_cls[label_cls == 0.] == False)
        if _tp+_fp == 0 or _tp+_fn==0 or _fp+_tn==0:
            continue
        precision[0,cls] = _tp/(_tp+_fp)
        recall[0,cls] = _tp/(_tp+_fn)
        fpr[0,cls] = _fp/(_fp+_tn)
    
    return precision, recall, fpr
        
def pr_curve(res, label):
    thresholds = np.linspace(0.01,0.99, 200)
    m_pr = np.zeros((1,3), dtype=np.float32)
    m_re = np.zeros((1,3), dtype=np.float32)
    m_fpr = np.zeros((1,3), dtype=np.float32)
    
    PR_thres_matrix = np.zeros((2,3,200),dtype=np.float32)
    ROC_thres_matrix = np.zeros((2,3,200),dtype=np.float32)
    for idx,thres in enumerate(thresholds):
        pr, re, fpr = cal_pr(res,label, thres)
        m_pr += pr
        m_re += re
        m_fpr += fpr
        PR = np.concatenate((pr,re), axis=0)
        ROC = np.concatenate((re,fpr), axis=0)
        PR_thres_matrix[:,:,idx]=PR
        ROC_thres_matrix[:,:,idx]=ROC
    m_pr /= len(thresholds)
    m_re /= len(thresholds)
    
    return PR_thres_matrix,ROC_thres_matrix, m_pr, m_re
    
def augmentor(img,label):
    rand_crop = random.random()
    if rand_crop <0.1:
        if random.random()<0.8:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
            label = label.transpose(Image.FLIP_LEFT_RIGHT)
        
        # if random.random()<0.5:
        #     img = img.transpose(Image.FLIP_TOP_BOTTOM)
        #     label = label.transpose(Image.FLIP_TOP_BOTTOM)
        
        # if random.random()<0.5: 
        #     img = img.transpose(Image.ROTATE_180)
        #     label = label.transpose(Image.ROTATE_180)
    elif rand_crop <0.4 and rand_crop >=0.2:
        img_big = img.resize((580,1024), Image.BILINEAR)
        label_big = label.resize((580,1024), Image.NEAREST)
         
        center_x = random.randint(240, 612)
        center_y = 240
         
        img = img_big.crop((center_x-240,center_y-240, center_x+240,center_y+240))
        label = label_big.crop((center_x-240,center_y-240, center_x+240,center_y+240))
    
    return img,label

class Coal_dataset(Dataset):

    def __init__(self, root_dir, transform = None, is_training = False):
        self.root_dir = root_dir   
        self.transform = transform 
        self.images = os.listdir(self.root_dir)
        self.is_training = is_training
    def __len__(self):
        return len(self.images)

    def __getitem__(self,index):
        image_index = self.images[index]
        img_path = os.path.join(self.root_dir, image_index)
        # img = cv2.imread(img_path)
        # img = img[:,:,::-1].copy()
        # img = Image.fromarray(np.uint8(img))
        img = Image.open(img_path)
        label = get_labels(img_path.replace("images", "labels").replace("jpg", "png"))
        
        if self.is_training == True:
            img, label = augmentor(img,label)
            
        if self.transform:
            img = self.transform(img)#对样本进行变换
        to_tensor = transforms.Compose([transforms.Resize((384,384), interpolation= InterpolationMode.NEAREST),transforms.PILToTensor()])
        label = to_tensor(label).float()
        
        sample = {'image':img,'label':label,'name':image_index}
        
        return sample #返回该样本

# test = get_labels(path)
# print(test.shape)
# print(test[0])
# pr_curve(np.ones((1,10,10,3)),np.zeros((1,10,10,3)))

# dataloader = Coal_dataset(path,transform= transforms.Compose([
#                                     transforms.ToTensor(),
#                                     transforms.Resize((480,480)), 
#                                     transforms.Normalize([0.2990, 0.2990, 0.2990], [0.1037, 0.1037, 0.1037])]), is_training= True )

# x = dataloader[0]