# S6 Assignment

This Repository contains the code for assignment 6 part 2 solution.
The Approach achieves **99.28% validation accuracy**, with **less than 20k parameters,** in **20 epochs**.

## Code Summary
Model Network
````
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 4, 3, padding=1) #input - 28 OUtput - 28 RF - 3
        self.bn1 = nn.BatchNorm2d(4)
        self.conv2 = nn.Conv2d(4, 16, 3, stride=2,padding=0) #input - 28 OUtput - 13 RF - 5
        self.bn2 = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 16, 3, padding=1) #input - 13 OUtput - 13 RF - 9
        self.bn3 = nn.BatchNorm2d(16)
        self.pool2 = nn.MaxPool2d(2, 2) #input - 13 OUtput - 6 RF - 13
        self.conv5 = nn.Conv2d(16, 32, 3, padding=1) #input - 6 OUtput - 6 RF - 21
        self.bn5 = nn.BatchNorm2d(32)
        self.conv6 = nn.Conv2d(32, 32, 3, stride=2,padding=1) #input - 6 OUtput - 3 RF - 29
        self.bn6 = nn.BatchNorm2d(32)
        self.conv7 = nn.Conv2d(32, 10, 3) #input - 3 OUtput - 1 RF - 45
        self.d_out = nn.Dropout2d(.25)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.bn1(x)
        x = self.d_out(x)
        x = F.relu(self.conv2(x))
        x = self.bn2(x)
        x = self.d_out(x)
        x = F.relu(self.conv3(x))
        x = self.bn3(x)
        x = self.pool2(x)
        x = self.d_out(x)
        x = F.relu(self.conv5(x))
        x = self.bn5(x)
        x = F.relu(self.conv6(x))
        x = F.relu(self.conv7(x))
        x = x.view(-1, 10)
        return F.log_softmax(x)

````
It contains 6 convolution layers, with one max pool layer.
The idea is to reach Receptive Field to full size of the input image (28),(which was achieved in layer 5), keeping low parameters.
````
#input - 1*28*28 OUtput - 4*28*28 RF - 3
#input - 4*28*28 OUtput - 8*13*13 RF - 5
#input - 8*13*13 OUtput - 16*13*13 RF - 9
#input - 16*13*13 OUtput - 16*6*6 RF - 13
#input - 16*6*6 OUtput - 32*6*6 RF - 21
#input - 32*6*6 OUtput - 32*3*3 RF - 29
#input - 32*3*3 OUtput - 10*1*1 RF - 45
````
Convolutional layers were accompnied by batch norm and drop out. 
Final output is given by log softmax function.

Model Summary
````
----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
================================================================
            Conv2d-1            [-1, 4, 28, 28]              40
       BatchNorm2d-2            [-1, 4, 28, 28]               8
         Dropout2d-3            [-1, 4, 28, 28]               0
            Conv2d-4           [-1, 16, 13, 13]             592
       BatchNorm2d-5           [-1, 16, 13, 13]              32
         Dropout2d-6           [-1, 16, 13, 13]               0
            Conv2d-7           [-1, 16, 13, 13]           2,320
       BatchNorm2d-8           [-1, 16, 13, 13]              32
         MaxPool2d-9             [-1, 16, 6, 6]               0
        Dropout2d-10             [-1, 16, 6, 6]               0
           Conv2d-11             [-1, 32, 6, 6]           4,640
      BatchNorm2d-12             [-1, 32, 6, 6]              64
           Conv2d-13             [-1, 32, 3, 3]           9,248
           Conv2d-14             [-1, 10, 1, 1]           2,890
================================================================
Total params: 19,866
Trainable params: 19,866
Non-trainable params: 0
----------------------------------------------------------------
````

Batch size 
Best model performance was observed with batch size of 64

Optimizer
SGD was used with learning rate 0.01 and momentum. It was observed that the validation accuracy was fluctuating b/w 98.8 to 99.1. Usind reducing lr helped to gain extrac .1% gain to 99.28
````
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min',patience=2,factor=.5)
````
