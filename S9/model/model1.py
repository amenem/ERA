import torch.nn as nn
import torch.nn.functional as F
class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.convbloc1 = nn.Sequential(
            nn.Conv2d(3, 64 , 3, stride=1, padding=1), # input fm 32  ,output fm 32, RF 3
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, 3, stride=1, padding=1), #out 32, RF 5
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 128, 3, stride=1, padding=1), #out 32, RF 7
            nn.ReLU(),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, 3, stride=2, padding=1), #out 16, RF 9
            nn.ReLU(),
            nn.BatchNorm2d(256),
        )
        self.convbloc2 = nn.Sequential(
            nn.Conv2d(256, 256 , 3, stride=1, padding=1), # out fm 16, RF 13
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 256, 3, stride=1, padding=1), #out 16, RF 17
            nn.ReLU(),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, 3, stride=1, padding=1), #out 16, RF 21
            nn.ReLU(),
            nn.BatchNorm2d(512),
            nn.Conv2d(512, 512, 3, stride=2, padding=1), #out 8, RF 25
            nn.ReLU(),
            nn.BatchNorm2d(512),
        )
        self.convbloc3 = nn.Sequential(
            nn.Conv2d(512, 1024 , 3, stride=1, padding=1), # output fm 8, RF 33
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 1024, 3, stride=1, padding=1), #out 8, RF 41
            nn.ReLU(),
            nn.BatchNorm2d(1024),
            nn.Conv2d(1024, 2048, 3, stride=1, padding=1), #out 8, RF 49
            # nn.Conv2d(16, 16, 3, stride=2, padding=1), #out 4, RF 57
        )

        self.gap = nn.AvgPool2d(8) # out 1*16
        self.fc  = nn.Linear(2048,10)

    def forward(self,x):
        x = self.convbloc1(x)
        x = self.convbloc2(x)
        x = self.convbloc3(x)
        x = self.gap(x)
        x = x.view(-1,2048)
        x = self.fc(x)
        x = F.log_softmax(x,dim=1)
        return x
    