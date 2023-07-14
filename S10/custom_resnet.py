import torch.nn as nn
import torch.nn.functional as F

class CustomRes(nn.Module):
    def __init__(self):
        super().__init__()
        self.prepLayer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=64, kernel_size=(3,3), stride=1, padding=1, bias=False),## in_feature_map = 32, out_feature_map = 32
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Layer1_X1 = nn.Sequential(
            nn.Conv2d(in_channels =64, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False), #out_feature_map = 32
            nn.MaxPool2d(2), #  out_feature_map = 16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.Layer1_R1 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False),#  out_feature_map = 16
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=(3,3), stride=1, padding=1, bias=False),#  out_feature_map = 16
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.Layer2 = nn.Sequential(
            nn.Conv2d(in_channels =128, out_channels=256, kernel_size=(3,3), stride=1, padding=1, bias=False), ##  out_feature_map = 16
            nn.MaxPool2d(2),##  out_feature_map = 8
            nn.BatchNorm2d(256),
            nn.ReLU()
        ) 
        self.Layer3_X2 = nn.Sequential(
            nn.Conv2d(in_channels =256, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False), #out_feature_map = 8
            nn.MaxPool2d(2), #  out_feature_map = 4
            nn.BatchNorm2d(512),
            nn.ReLU()
        )
        self.Layer3_R2 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False),#  out_feature_map = 4
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=(3,3), stride=1, padding=1, bias=False),#  out_feature_map = 4
            nn.BatchNorm2d(512),
            nn.ReLU()
        )   
        self.pool = nn.MaxPool2d(4) #  out_feature_map = 1
        self.fc = nn.Linear(512, 10)

    def forward(self,X):
        X = self.prepLayer(X)
        X = self.Layer1_X1(X)
        R1 = self.Layer1_R1(X)
        X = R1+X
        X = self.Layer2(X)
        X = self.Layer3_X2(X)
        R2 = self.Layer3_R2(X)
        X = R2+X
        X = self.pool(X)
        X = X.view(-1,512)
        X = self.fc(X)
        X = F.log_softmax(X,dim=-1)
        return X

