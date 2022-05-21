import streamlit as st
import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2
import pandas as pd
from io import BytesIO, StringIO
import numpy as np
from PIL import Image

st.title("Digit Recognizer")

class ConvBlock(nn.Module):
    def __init__(self, multiplyer, kernel_size):
        super().__init__()
        in_c = 1 if multiplyer == 1 else 32
        padding =  2 if multiplyer == 1 else 1
        self.conv1 = nn.Conv2d(in_c, 32*multiplyer, kernel_size, 1, padding)
        self.conv2 = nn.Conv2d(32*multiplyer, 32*multiplyer, kernel_size, 1, padding)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=0.25)
        self.maxpool = nn.MaxPool2d(2)
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.maxpool(x)
        x = self.dropout(x)

        return x

class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.convblock1 = ConvBlock(1, 5)
        self.convblock2 = ConvBlock(2, 3)
        self.fc1 = nn.Linear(64 * 7 * 7, 256)
        self.dropout = nn.Dropout(p=0.5)
        self.fc2 = nn.Linear(256, 10)
    
    def forward(self, x):
        x = self.convblock1(x)
        x = self.convblock2(x)

        out = self.fc1(x.flatten(1))
        out = self.dropout(out)
        out = self.fc2(out)
        
        return out

data = st.sidebar.file_uploader("Load image", ['png', 'jpg', 'jpeg'])
if data is not None:
    model = CNN()
    model.load_state_dict(torch.load("model.pth", map_location="cpu"))
    model.eval()
    image = Image.open(data).convert('RGB')
    image = np.array(image)
    st.image(cv2.resize(image, (500, 500)))
    image = image[:, :, ::-1]
    image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    image = cv2.resize(image, (28, 28)).reshape(-1, 1, 28, 28) / 255
    img = torch.tensor(image, dtype=torch.float32)
    res = (model(img).max(1, keepdim=True)[1]).numpy()
    st.title("Digit in image is: {}".format(int(np.mean(res))))