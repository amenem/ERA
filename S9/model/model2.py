import torch.nn as nn
import torch.nn.functional as F
class Net2(nn.Module):
    def __init__(self):
        super().__init__()

        self.convbloc1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=1, padding=1), # input fm 32  ,output fm 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 32, 3, stride=1, padding=1), #out 32, RF 5
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), #out 32, RF 7
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, 3, stride=2, padding=1, dilation=2), #out 15, RF 9
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.convbloc2 = nn.Sequential(
            nn.Conv2d(16, 32 , 3, stride=1, padding=1), # out fm 15, RF 13
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), #out 15, RF 17
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), #out 15, RF 21
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 16, 3, stride=2, padding=1, dilation=2), #out 7, RF 25
            nn.ReLU(),
            nn.BatchNorm2d(16),
        )
        self.convbloc3 = nn.Sequential(
            nn.Conv2d(16, 32 , 3, stride=1, padding=1), # output fm 7, RF 33
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, 3, stride=1, padding=1), #out 7, RF 41
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 64, 3, stride=1, padding=1), #out 7, RF 49
        )

        self.gap = nn.AvgPool2d(7) # out 1*16
        self.fc  = nn.Linear(64,10)

    def forward(self,x):
        x = self.convbloc1(x)
        # print(x.shape)
        x = self.convbloc2(x)
        # print(x.shape)
        x = self.convbloc3(x)
        # print(x.shape)
        x = self.gap(x)
        x = x.view(-1,64)
        x = self.fc(x)
        x = F.log_softmax(x,dim=1)
        return x
    