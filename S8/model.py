import torch
import torch.nn as nn
import torch.nn.functional as F

class BN(nn.Module):
    def __init__(self):
        super().__init__()

        self.convbloc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3), #30
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3), # 28
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3), # 26
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.t_layer1 = nn.Sequential(
            nn.MaxPool2d(2) # 13
        )
        self.convbloc2 = nn.Sequential(
            nn.Conv2d(16, 16, 3), #11
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3), # 9
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, 3), # 7
            nn.ReLU(),
            nn.BatchNorm2d(16)
        )
        self.convbloc3 = nn.Sequential(
            nn.Conv2d(16,10,1), #5
            nn.AvgPool2d(5)
        )
    def forward(self,x):
        x = self.convbloc1(x)
        x = self.t_layer1(x)
        x = self.convbloc2(x)
        x = self.convbloc3(x)
        x = x.view(-1,10)
        x = F.log_softmax(x)
        return x

