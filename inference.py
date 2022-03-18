import numpy as np
import cv2
import torch
import torchvision
import torch.nn as nn
import os
import sys
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from model import Res34net,Res18_segNet
from utils import Coal_dataset, cal_pr, pr_curve, addPRendpoint
import matplotlib.pyplot as plt

COLORS = [(0, 0, 0), (0, 0, 255), (0, 255, 0) ]

def detect(img_path):
    model_path = "./res18_seg_last.pth"
    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    print(device)
    img = cv2.imread(img_path+"201.jpg")
    print(img.shape)
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Resize((480,480)),
                                    transforms.Normalize([0.2990, 0.2990, 0.2990], [0.1037, 0.1037, 0.1037])     #train
                                    ])

    dataset = Coal_dataset(img_path,transform = transform, is_training=False)
    dataloader = DataLoader(dataset)
    
    model = torch.load(model_path)
    model.eval()
    model.to(device)
    # print(model)
    
    m_pr_0595 = 0.0
    m_re_0595 = 0.0
    PR_sum = np.zeros((2,3,200),dtype=np.float32)
    ROC_sum = np.zeros((2,3,200),dtype=np.float32)
    for idx,data in enumerate(dataloader) :
        img_input = data["image"].to(device)
        res = model(img_input)
        result = res[0,:,:,:]
        # print(result.shape)        
        result_copy = result.cpu().detach().numpy()
        result_idx = np.argmax(result_copy, axis=0)
        # print(result_idx.shape)
        seg_res = np.zeros((384,384,3),dtype=np.uint8)
        for i in np.unique(result_idx):
            seg_res[result_idx == i] = COLORS[i]
        seg_res = cv2.resize(seg_res,(img.shape[1], img.shape[0]))
        front = np.sum(result_idx !=0)
        coal = np.sum(result_idx == 1)
        coal_percentage = coal/front
        cv2.putText(seg_res, "coal:"+'%.3f'%coal_percentage,(20,100), fontFace= cv2.FONT_HERSHEY_COMPLEX, fontScale=0.75,color=(0,0,255), thickness= 1)
        cv2.imwrite("results_train/"+data["name"][0],seg_res)
        
        res_ = res.cpu().detach().numpy()
        lb_ = data["label"].cpu().detach().numpy()
        PR, ROC, m_pr, m_re = pr_curve(res_,lb_)
        m_pr_0595 += m_pr
        m_re_0595 += m_re
        PR_sum += PR
        ROC_sum += ROC
        
    m_pr_0595 /= len(dataloader)
    m_re_0595 /= len(dataloader)
    PR_mean  = PR_sum/len(dataloader)
    ROC_mean = ROC_sum/len(dataloader)
    
    bg_pre_list =  PR_mean[0,0,:].tolist()
    bg_re_list = PR_mean[1,0,:].tolist()
    bg_pre_list, bg_re_list = addPRendpoint(bg_pre_list, bg_re_list)
    
    coal_pre_list = PR_mean[0,1,:].tolist()
    coal_re_list = PR_mean[1,1,:].tolist()
    coal_pre_list, coal_re_list = addPRendpoint(coal_pre_list, coal_re_list)
  
    gauge_pre_list = PR_mean[0,2,:].tolist()
    gauge_re_list = PR_mean[1,2,:].tolist()
    gauge_pre_list, gauge_re_list = addPRendpoint(gauge_pre_list, gauge_re_list)

    overall_pre_list = (np.sum(PR_mean,axis=1)/3)[0,:].tolist()
    overall_re_list = (np.sum(PR_mean,axis=1)/3)[1,:].tolist()
    overall_pre_list, overall_re_list = addPRendpoint(overall_pre_list, overall_re_list)

    fig,ax = plt.subplots()
    ax.plot(bg_re_list,bg_pre_list,label ="background")
    ax.plot(coal_re_list,coal_pre_list,label ="coal")
    ax.plot(gauge_re_list,gauge_pre_list,label="gangue")
    ax.plot(overall_re_list,overall_pre_list,label="all class")
    ax.set_xlabel("recall")
    ax.set_ylabel("precision")
    ax.legend()
    plt.savefig("./PR_curve.jpg")
    
    bg_tpr_list =  ROC_mean[0,0,:].tolist()
    bg_fpr_list = ROC_mean[1,0,:].tolist()
    bg_tpr_list.append(0.0)
    bg_fpr_list.append(0.0)
    bg_tpr_list.insert(0,1.0)
    bg_fpr_list.insert(0,1.0)
    
    coal_tpr_list = ROC_mean[0,1,:].tolist()
    coal_fpr_list = ROC_mean[1,1,:].tolist()
    coal_tpr_list.append(0.0)
    coal_fpr_list.append(0.0)
    coal_tpr_list.insert(0,1.0)
    coal_fpr_list.insert(0,1.0) 
    
    gauge_tpr_list = ROC_mean[0,2,:].tolist()
    gauge_fpr_list = ROC_mean[1,2,:].tolist()
    gauge_tpr_list.append(0.0)
    gauge_fpr_list.append(0.0)
    gauge_tpr_list.insert(0,1.0)
    gauge_fpr_list.insert(0,1.0)

    overall_tpr_list = (np.sum(ROC_mean,axis=1)/3)[0,:].tolist()
    overall_fpr_list = (np.sum(ROC_mean,axis=1)/3)[1,:].tolist()
    overall_tpr_list.append(0.0)
    overall_fpr_list.append(0.0)
    overall_tpr_list.insert(0,1.0)
    overall_fpr_list.insert(0,1.0)
    
    fig2,ax2 = plt.subplots()
    ax2.plot(bg_fpr_list,bg_tpr_list,label ="background")
    ax2.plot(coal_fpr_list,coal_tpr_list,label ="coal")
    ax2.plot(gauge_fpr_list,gauge_tpr_list,label="gangue")
    ax2.plot(overall_fpr_list,overall_tpr_list,label="all class")
    ax2.set_xlabel("FPR")
    ax2.set_ylabel("TPR")
    ax2.legend()
    plt.savefig("./ROC_curve.jpg")
    
    
        
if __name__=="__main__":
    detect("./train_data/images/")