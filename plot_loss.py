import os
import matplotlib.pyplot as plt

from train import train

def plot_curve(path):
    file = open(path, "r")
    lines = file.readlines()
    idx = []
    train_loss = []
    val_loss = []
    acc = []
    lr = []
    for line in lines:
        epc = line.split(" ")
        if epc[0] == "[epoch":
            idx.append(int(epc[1].split("]")[0]))
            data = line.split("]")[1].split("   ")[0].strip("\n")
            data = data.split("  ")
            data[0] = data[0].strip()
            # dic_data[int(idx)] = data
            train_loss.append(float(data[0].split(" ")[1].strip()))
            val_loss.append(float(data[2].split(" ")[1].strip()))
            acc.append(float(data[1].split(" ")[1].strip()))
            # lr.append(float(data[3].split(" ")[1].strip()))

            
        else:
            continue
    
    fig = plt.figure(figsize=(15,5))
    
    ax_train_loss = fig.add_subplot(1,3,1)
    ax_train_loss.set_title("train loss")
    ax_train_loss.plot(idx,train_loss)
    
    ax_acc = fig.add_subplot(1,3,2)
    ax_acc.set_title("val accurancy")
    ax_acc.plot(idx,acc)
    
    ax_val_loss = fig.add_subplot(1,3,3)
    ax_val_loss.set_title("val loss")
    ax_val_loss.plot(idx,val_loss)
    
    plt.subplots_adjust(wspace=0.5)
    
    plt.savefig("./figure.jpg")
    
    

            
            

if __name__ == "__main__":
    plot_curve("./nohup.out")

