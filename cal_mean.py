import numpy as np
import cv2
import os
def main(path):
    files = os.listdir(path)
    R = 0
    G = 0
    B = 0
    R_sq = 0
    G_sq = 0
    B_sq = 0

    std_R = 0
    std_G = 0
    std_B = 0
    num = len(files)
    pix_num = 0
    print(num)
    for img_name in files:
        
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        # print(img)
        # cv2.imshow("test",img)
        # cv2.waitKey(0)
        img = img[:,:,::-1]
        pix_num += img.shape[0]*img.shape[1]

        R_temp = img[:,:,0]
        R += np.sum(R_temp)
        
        G_temp = img[:,:,1]
        G += np.sum(G_temp)

        B_temp = img[:,:,2]
        B += np.sum(B_temp)

        R_sq += np.sum(np.power(R_temp,2.0))
        G_sq += np.sum(np.power(G_temp,2.0))
        B_sq += np.sum(np.power(B_temp,2.0))


    mean = np.array((R, G, B))/pix_num
    std_R = np.sqrt(R_sq/pix_num-mean[0]**2)
    std_G = np.sqrt(R_sq/pix_num-mean[1]**2)
    std_B = np.sqrt(R_sq/pix_num-mean[2]**2)
    std = np.array((std_R, std_G, std_B))/255

    print(mean)
    print(std)

if __name__ =="__main__":
    main("../raw_data")