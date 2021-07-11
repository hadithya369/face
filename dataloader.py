import numpy as np
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from skimage import io
import os
import random
import matplotlib.pyplot as plt
from torchvision.io import read_image

class LFWDataset(Dataset):
    def __init__(self, namefile, direct, transform=None, target_transform=None):
        self.direct=direct
        self.transform=transform
        self.target_transform=target_transform
        self.namefile=open(namefile,'r').readlines()
        self.namefile=[el.split() for el in self.namefile]
    
    def __len__(self):
        return len(self.namefile)
    
    def getimgpath(self,el0,el1):
        img1_name=el0+'_{:04d}'.format(int(el1)+1)+'.jpg'
        img1_path=os.path.join(self.direct,el0)
        img1_path=os.path.join(img1_path,img1_name)
        return img1_path
    
    def __getitem__(self,idx):
        img1_path=self.getimgpath(self.namefile[idx][0],self.namefile[idx][1])
        img2_path=self.getimgpath(self.namefile[idx][2],self.namefile[idx][3])        
        img1=read_image(img1_path)
        img2=read_image(img2_path)
        label=int(self.namefile[idx][4])
        if self.transform:
            img1=self.transform(img1)
            img2=self.transform(img2)
        if self.target_transform:
            label=self.target_transform(label)
        return img1,img2,label

