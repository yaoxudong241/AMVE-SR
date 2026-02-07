import cv2
import os
from tqdm import tqdm

LRimagePath = r"D:\code\ESRT-RS\npydata\AID\test\LR_bicubic\X{}/"
HRimagePath = r"D:\code\ESRT-RS\npydata\AID\test\HR/"

targetLRimagePath =r"D:\code\ESRT-RS\npydata\AID\test\testforCAM\LR\X{}/"
targetHRimagePath = r"D:\code\ESRT-RS\npydata\AID\test\testforCAM\HR/"

scale = [2,4,8]
size = 512

for root, dirs, files in os.walk(HRimagePath):
    for file in tqdm(files):
        if file.endswith('.png'):
            HRimage = cv2.imread(HRimagePath+file)
            cv2.imwrite(targetHRimagePath+file, HRimage[0:size,0:size,:])
            for i in range (len(scale)):
                LRimage = cv2.imread(LRimagePath.format(scale[i])+file)
                cv2.imwrite(targetLRimagePath.format(scale[i]) + file, LRimage[0:int(size/scale[i]), 0:int(size/scale[i]), :])


