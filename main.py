import cv2
import torch
import numpy as np
from torchvision.transforms import ToTensor
from torchvision import transforms
from torch.utils.data import Dataset,DataLoader
import pandas as pd
from skimage import io
import os
import random
import matplotlib.pyplot as plt
from dataloader import LFWDataset
from face_model import FaceRecognition
import datetime
from torchvision.io import read_image

# Load the cascade
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model=FaceRecognition().to(device)
model.load_state_dict(torch.load('mode39',map_location=torch.device('cpu')))
loss_fn = torch.nn.MSELoss()


# To capture video from webcam. 
cap = cv2.VideoCapture(0)

def validate(img):
    img = np.array([img[:,:,2],img[:,:,1],img[:,:,0]])
    p = transforms.Compose([transforms.Scale((250,250))])
    new_amitabh = p(read_image('amitabh.jpg'))
    new_amitabh2 = p(read_image('amitabh2.jpg'))
    new_img = torch.from_numpy(img)
    new_img = p(new_img)
    new_amitabh = torch.unsqueeze(new_amitabh, 0)
    new_amitabh2 = torch.unsqueeze(new_amitabh2, 0)
    new_img = torch.unsqueeze(new_img, 0)
    model.eval()
    with torch.no_grad():
        pred = model(new_amitabh/255.0, new_amitabh2/255.0)
    print(pred)

while True:
    _, img = cap.read()
    # print(img.shape)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    if(len(faces) > 1):
        faces = [faces[0]]
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
        validate(img[x:x+h,y:y+w,:])
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()