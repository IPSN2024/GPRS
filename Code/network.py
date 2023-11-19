import torch
import torch.nn as nn
import torch.nn.functional as F


class RadioTomoNet(nn.Module):
    def __init__(self,in_channels) -> None:
        super(RadioTomoNet,self).__init__()
        self.in_channels = in_channels
        self.conv1 = nn.Conv2d(in_channels=in_channels,out_channels=128,kernel_size=(3,3),stride=1)
        self.bn1 = nn.BatchNorm2d(128)
        self.conv2 = nn.Conv2d(in_channels=128,out_channels=512,kernel_size=(4,4),stride=1)
        self.bn2 = nn.BatchNorm2d(512)
        self.conv3 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(5,5),stride=1)
        self.bn3 = nn.BatchNorm2d(1024)
        self.conv4 = nn.Conv2d(in_channels=1024,out_channels=2048,kernel_size=(7,7),stride=1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.relu = nn.ReLU(inplace=True)

        self.pathconv1 = nn.Conv2d(in_channels=in_channels,out_channels=512,kernel_size=(9,9),stride=1)
        self.pathbn1 = nn.BatchNorm2d(512)
        self.pathconv2 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(7,7),stride=1)
        self.pathbn2 = nn.BatchNorm2d(1024)
        self.pathconv3 = nn.Conv2d(in_channels=1024,out_channels=2048,kernel_size=(2,2),stride=1)
        self.pathbn3 = nn.BatchNorm2d(2048)
        
        #假设是相同的Block
        self.path1conv1 = nn.Conv2d(in_channels=in_channels,out_channels=128,kernel_size=(2,2))
        self.path1bn1 = nn.BatchNorm2d(128)
        self.path1conv2 = nn.Conv2d(in_channels=128,out_channels=512,kernel_size=(5,5))
        self.path1bn2 = nn.BatchNorm2d(512)
        self.path1conv3 = nn.Conv2d(in_channels=512,out_channels=1024,kernel_size=(5,5))
        self.path1bn3 = nn.BatchNorm2d(1024)
        self.path1conv4 = nn.Conv2d(in_channels=1024,out_channels=2048,kernel_size=(5,5))
        self.path1bn4 = nn.BatchNorm2d(2048)
        
        self.dropout = nn.Dropout()
        self.Fc1 = nn.Linear(2048,3000)
        self.Fc2 = nn.Linear(3000,3600)
    def forward(self,X):
        res1 = self.relu(self.bn1(self.conv1(X)))
        res1 = self.relu(self.bn2(self.conv2(res1)))
        res1 = self.relu(self.bn3(self.conv3(res1)))
        res1 = self.bn4(self.conv4(res1))

        res2 = self.relu(self.pathbn1(self.pathconv1(X)))
        res2 = self.pathbn2(self.pathconv2(res2))
        res2 = self.pathbn3(self.pathconv3(res2))

        res3 = self.relu(self.path1bn1(self.path1conv1(X)))
        res3 = self.relu(self.path1bn2(self.path1conv2(res3)))
        res3 = self.relu(self.path1bn3(self.path1conv3(res3)))
        res3 = self.path1bn4(self.path1conv4(res3))

        #print(res1.shape,res2.shape,res3.shape)

        res = res1 + res2 + res3
        #res = res1
        res = torch.flatten(res,start_dim=1)
        res = self.dropout(res)
        res = self.relu(self.Fc1(res))
        res = self.Fc2(res)
        res = res.reshape(-1,60,60)
        return res


if __name__ == "__main__":
    data = torch.rand(size=[16,16,16,16])
    mo = RadioTomoNet(in_channels=16)
    res = mo(data)
    print(res.shape)
    
    
