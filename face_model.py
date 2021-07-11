import torch
from torch import nn
import torch.nn.functional as F

class FaceRecognition(nn.Module):
    def __init__(self):
        super(FaceRecognition,self).__init__()
        self.conv1=nn.Conv2d(3,32,3)
        self.bn1=nn.BatchNorm2d(32)
        self.conv2=nn.Conv2d(32,32,3)
        self.bn2=nn.BatchNorm2d(32)
        self.conv3=nn.Conv2d(32,64,3)
        self.bn3=nn.BatchNorm2d(64)
        self.conv4=nn.Conv2d(64,128,3)
        self.bn4=nn.BatchNorm2d(128)
        self.conv5=nn.Conv2d(128,128,3)
        self.bn5=nn.BatchNorm2d(128)
        self.conv6=nn.Conv2d(128,128,3)
        self.bn6=nn.BatchNorm2d(128)
        self.conv7=nn.Conv2d(128,256,3)
        self.bn7=nn.BatchNorm2d(256)
        self.conv8=nn.Conv2d(256,256,3)
        self.bn8=nn.BatchNorm2d(256)
        self.conv9=nn.Conv2d(256,512,3)
        self.bn9=nn.BatchNorm2d(512)
        self.conv10=nn.Conv2d(512,512,3)
        self.bn10=nn.BatchNorm2d(512)
        self.conv11=nn.Conv2d(512,1024,3)
        self.bn11=nn.BatchNorm2d(1024)
        self.fc0=nn.Linear(1024,256)
        self.drop1=nn.Dropout(p=0.2)
        self.fc1=nn.Linear(256,64)
        self.drop2=nn.Dropout(p=0.2)
        self.fc2=nn.Linear(64,16)
        self.fc3=nn.Linear(16,1)
        
    def siam(self,x):
        # 3,250,250
        x=F.relu(self.bn1(self.conv1(x)))
        # 32,248,248
        x=F.relu(self.bn2(self.conv2(x)))
        # 32,246,246
        x=F.max_pool2d(x,2)
        # 32,123,123
        x=F.relu(self.bn3(self.conv3(x)))
        # 64,121,121
        x=F.max_pool2d(x,2)
        # 64,60,60
        x=F.relu(self.bn4(self.conv4(x)))
        # 128,58,58
        x=F.relu(self.bn5(self.conv5(x)))
        # 128,56,56
        x=F.relu(self.bn6(self.conv6(x)))
        # 128,54,54
        x=F.max_pool2d(x,2)
        # 128,27,27
        x=F.relu(self.bn7(self.conv7(x)))
        # 256,25,25
        x=F.max_pool2d(x,2)
        # 256,12,12
        return x
    
    def forward(self,x1,x2):
        x1=self.siam(x1)
        x2=self.siam(x2)
        x=torch.abs(torch.sub(x1,x2))
        # 256,12,12
        x=F.relu(self.bn8(self.conv8(x)))
        # 256,10,10
        x=F.relu(self.bn9(self.conv9(x)))
        # 512,8,8
        x=F.relu(self.bn10(self.conv10(x)))
        # 512,6,6
        x=F.max_pool2d(x,2)
        # 512,3,3
        x=F.relu(self.bn11(self.conv11(x)))
        # 1024,1,1
        x=x.view(-1,1024)
        x=F.relu(self.fc0(x))
        x=self.drop1(x)
        x=F.relu(self.fc1(x))
        x=self.drop2(x)
        x=F.relu(self.fc2(x))
        x=torch.sigmoid(self.fc3(x))
        return x

