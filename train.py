from cv2 import imshow
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
                                    transforms.ToTensor(),
                                    transforms.RandomHorizontalFlip(),
                                    transforms.Resize((224,224)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                                    ]),

        "val": transforms.Compose([
                                    transforms.ToTensor(),
                                    transforms.Resize((224,224)), 
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    }
    
    images_path = "./train_data/images/"
    train_dataset = Coal_dataset(root_dir= images_path, transform= data_transform["train"])
    train_data_num = len(train_dataset)
    
    val_images_path = "./test_data/images/"
    val_dataset = Coal_dataset(root_dir= images_path, transform= data_transform["val"])
    val_data_num = len(val_dataset)
    # test=train_dataset[1]
    # cv2.imshow("test",test["image"])
    # cv2.waitKey(0)
    # print(test["label"])

    # print(train_data_num)
    # input data size is 852*480
    
    model = Res18_segNet()
    model.to(device)
    # y = model(torch.randn(4,3,224,224).to(device))
    # print(y.size())
    batch_size = 4
    # nw = min([batch_size if batch_size > 1 else 0, 8])  # number of workers
    train_loader = DataLoader(train_dataset, batch_size= batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size= batch_size, shuffle=True, num_workers=0)
    print("using {} images for training, {} images for validation.".format(train_data_num, val_data_num))
    
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr = 0.001)
    
    epochs = 50
    best_acc = 0.0
    save_path = "./res18_seg.pth"
    train_steps = len(train_loader)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, file= sys.stdout)
        for step, data in enumerate(train_loader):
            images, labels = data["image"].to(device), data["label"].to(device)
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
                # loss = loss_function(outputs, test_labels)
                predict_y = torch.argmax(outputs,dim=1)
                label_y = torch.argmax(val_labels.to(device), dim= 1)
                acc += torch.eq(predict_y, label_y).sum().item()

                val_bar.desc = "valid epoch[{}/{}]".format(epoch + 1,
                                                           epochs)

        val_accurate = acc / val_data_num
        print('[epoch %d] train_loss: %.3f  val_accuracy: %.3f' %
              (epoch + 1, running_loss / train_steps, val_accurate))

        if val_accurate > best_acc:
            best_acc = val_accurate
            torch.save(model.state_dict(), save_path)

    print('Finished Training')
            

        
        



if __name__ == "__main__":
    train()