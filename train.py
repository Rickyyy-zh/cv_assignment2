from math import gamma
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
from utils import Coal_dataset, get_labels
from model import Res34net,Res18_segNet


from torchvision import transforms

def train():
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
    print("The device is ", device)

    data_transform = {
        "train": transforms.Compose([
                                    # transforms.RandAugment(num_ops=4),
                                    transforms.ToTensor(),
                                    # transforms.RandomHorizontalFlip(),
                                    transforms.Resize((480,480)),
                                    transforms.Normalize([0.2990, 0.2990, 0.2990], [0.1037, 0.1037, 0.1037])
                                    ]),

        "val": transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((480,480)), 
                                    transforms.Normalize([0.2990, 0.2990, 0.2990], [0.1037, 0.1037, 0.1037])])
    }
    
    images_path = "./train_data/images/"
    train_dataset = Coal_dataset(root_dir= images_path, transform= data_transform["train"],is_training = True)
    train_data_num = len(train_dataset)
    
    val_images_path = "./test_data/images/"
    val_dataset = Coal_dataset(root_dir= val_images_path, transform= data_transform["val"],is_training = False)
    val_data_num = len(val_dataset)
    
    model = Res18_segNet()
    model.to(device)
    y = model(torch.randn(4,3,480,480).to(device))
    print(y.size())
    batch_size = 4
    # nw = min([batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=True, num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_data_num, val_data_num))
    
    loss_fn = nn.CrossEntropyLoss()
    # loss_fn = FocalLossV2()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001, momentum=0.99, weight_decay=0.0005)
    step_lr = torch.optim.lr_scheduler.MultiStepLR(optimizer=optimizer,milestones=[2,20,50,100,200], gamma=0.5)
    epochs = 300
    best_acc = 0.0
    save_path = "./res18_seg_best.pth"
    save_path_last = "./res18_seg_last.pth"
    
    train_steps = len(train_loader)
    val_steps = len(val_loader)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        running_loss_val = 0.0
        train_bar = tqdm(train_loader, file= sys.stdout)
        for step, data in enumerate(train_loader):
            images, labels = data["image"].to(device), data["label"].to(device)
            # labels = torch.argmax(labels,dim=1)
            optimizer.zero_grad()
            logits = model(images)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.desc = "train epoch [{}/{}]  loss:{:.3f}".format(epoch+1, epochs, loss)
        
        model.eval()
        acc = 0.0
        with torch.no_grad():
            val_bar = tqdm(val_loader, file=sys.stdout)
            for val_data in val_loader:
                val_images, val_labels = val_data["image"].to(device), val_data["label"].to(device)
                
                outputs = model(val_images)
                label_y = torch.argmax(val_labels.to(device), dim= 1)
                val_loss = loss_fn(outputs, val_labels)
                predict_y = torch.argmax(outputs,dim=1)
                acc += torch.eq(predict_y, label_y).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)
                running_loss_val += val_loss.item()

        val_accurate = acc / val_data_num /(384*384)
        learn_rate = step_lr.get_last_lr()
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f  val_loss: %.3f  lr: %f' %
              (epoch + 1, running_loss / train_steps, val_accurate, running_loss_val/val_steps, learn_rate[0]))
        step_lr.step()

        if val_accurate >= best_acc:
            best_acc = val_accurate
            torch.save(model, save_path)
        torch.save(model, save_path_last)
        

    print('Finished Training')
            
if __name__ == "__main__":
    train()