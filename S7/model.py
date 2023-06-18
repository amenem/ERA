import torch.nn as nn
import torch.nn.functional as F

dropout_value = 0.1

class Model_1(nn.Module):
    def __init__(self):
        super().__init__()

        self.convblock1 = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=16 ,kernel_size=(3,3),padding=0, bias=False),# 26 input/channel size
          nn.ReLU(),
          nn.Conv2d(in_channels=16, out_channels=32 ,kernel_size=(3,3),padding=0, bias=False), ## 24
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64 ,kernel_size=(3,3),padding=0, bias=False), ## 22
          nn.ReLU(),
        )
        self.pool1 = nn.Sequential( ## 11
          nn.MaxPool2d(2,2),
        )
        self.convblock2 = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=16 ,kernel_size=(3,3),padding=1, bias=False), ##  11
          nn.ReLU(),
          nn.Conv2d(in_channels=16, out_channels=32,kernel_size=(3,3),padding=1, bias=False), ##  11
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64 ,kernel_size=(3,3),padding=0, bias=False), ##  9
        )
        self.convblock3 = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=16 ,kernel_size=(3,3),padding=1, bias=False), ##  9
          nn.ReLU(),
          nn.Conv2d(in_channels=16, out_channels=32 ,kernel_size=(3,3),padding=1, bias=False), ## 9
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=64 ,kernel_size=(3,3),padding=0, bias=False), ## 7
          nn.ReLU(),
        )
        self.pool2 = nn.Sequential( ## 3
          nn.MaxPool2d(2,2),
        )
        self.convblock4 = nn.Sequential(
          nn.Conv2d(in_channels=64, out_channels=16 ,kernel_size=3,padding=1, bias=False), ##, 3
          nn.ReLU(),
          nn.Conv2d(in_channels=16, out_channels=32 ,kernel_size=3,padding=1, bias=False), ## 3
          nn.ReLU(),
          nn.Conv2d(in_channels=32, out_channels=10 ,kernel_size=3,padding=0, bias=False), ## 1
          # nn.ReLU()
        )

    def forward(self, x):
        x = self.convblock1(x)
        # print (x.shape)
        x = self.pool1(x)
        # print (x.shape)
        x= self.convblock2(x)
        # print (x.shape)
        x= self.convblock3(x)
        # print (x.shape)
        x= self.pool2(x)
        # print (x.shape)
        x= self.convblock4(x)
        # print (x.shape)
        x = x.view(-1,10)
        x = F.log_softmax(x)
        return x 

class Model_2(nn.Module):
    def __init__(self):
        super().__init__()

        self.convblock1 = nn.Sequential(
          nn.Conv2d(in_channels=1, out_channels=2 ,kernel_size=(3,3),padding=0, bias=False),
          nn.ReLU(),
          nn.Conv2d(in_channels=2, out_channels=4 ,kernel_size=(3,3),padding=0, bias=False), 
          nn.ReLU(),
          nn.Conv2d(in_channels=4, out_channels=8 ,kernel_size=(3,3),padding=0, bias=False), 
          nn.ReLU(),
        )
        self.pool1 = nn.Sequential( 
          nn.MaxPool2d(2,2),
        )
        self.convblock2 = nn.Sequential(
          nn.Conv2d(in_channels=8, out_channels=2 ,kernel_size=(3,3),padding=1, bias=False), 
          nn.ReLU(),
          nn.Conv2d(in_channels=2, out_channels=4 ,kernel_size=(3,3),padding=1, bias=False), 
          nn.ReLU(),
          nn.Conv2d(in_channels=4, out_channels=8 ,kernel_size=(3,3),padding=0, bias=False), 
          nn.ReLU(),        
        )
        self.convblock3 = nn.Sequential(
          nn.Conv2d(in_channels=8, out_channels=2 ,kernel_size=(3,3),padding=1, bias=False),
          nn.ReLU(),
          nn.Conv2d(in_channels=2, out_channels=4 ,kernel_size=(3,3),padding=1, bias=False), 
          nn.ReLU(),
          nn.Conv2d(in_channels=4, out_channels=8 ,kernel_size=(3,3),padding=0, bias=False),
          nn.ReLU(),
        )
        self.pool2 = nn.Sequential(
          nn.MaxPool2d(2,2),
        )
        self.convblock4 = nn.Sequential(
          nn.Conv2d(in_channels=8, out_channels=8 ,kernel_size=3,padding=1, bias=False),
          nn.ReLU(),
          nn.Conv2d(in_channels=8, out_channels=8 ,kernel_size=3,padding=1, bias=False), 
          nn.ReLU(),
          nn.Conv2d(in_channels=8, out_channels=10 ,kernel_size=3,padding=0, bias=False),
          # nn.ReLU()
        )

    def forward(self, x):
        x = self.convblock1(x)
        # print (x.shape)
        x = self.pool1(x)
        # print (x.shape)
        x= self.convblock2(x)
        # print (x.shape)
        x= self.convblock3(x)
        # print (x.shape)
        x= self.pool2(x)
        # print (x.shape)
        x= self.convblock4(x)
        # print (x.shape)
        x = x.view(-1,10)
        x = F.log_softmax(x)
        return x 
