#encoding=utf-8

from UNetUtils import *

class UNetModel(nn.Module):
    def __init__(self,n_channels,bilinear=True) -> None:
        super(UNetModel,self).__init__()
        self.n_channels = n_channels
        self.bilinear = bilinear

        self.inc = DoubleConv(n_channels,64)
        self.down1 = Down(64,128)
        self.down2 = Down(128,256)
        #self.down3 = Down(256,512)
        #self.down4 = Down(512,1024)

        #self.up1 = UpTransPose(1024,512)
        #self.up2 = UpTransPose(512,256,kernel_size=2,stride=2)
        self.up3 = UpTransPose(256,128,kernel_size=2,stride=2)
        #self.conv1 = nn.Conv2d(128,64,kernel_size=(5,5),stride=1)
        #self.conv2 = nn.Conv2d(64,128,kernel_size=(5,5),stride=1)
        self.up4 = UpTransPose(128,64,kernel_size=2,stride=2)
        self.ouputlayer = nn.Conv2d(in_channels=64,out_channels=1,kernel_size=(1,1),stride=1)
        #self.relu = nn.ReLU()

    def forward(self,x):
        #encoding
        x1 = self.inc(x)
        x2 = self.down1(x1) 
        x3 = self.down2(x2)
        #x4 = self.down3(x3)
        #x5 = self.down4(x4)
        #print("the x shape is: ",x3.shape)

        #Deconding 
        #x = self.up1(x5,x4)
        #x = self.up2(x,x3)
        x = self.up3(x3,x2)
        #x = self.conv1(x) 
        #x = self.conv2(x)
        x = self.up4(x,x1)
        out = self.ouputlayer(x) 
        return out

if __name__ == "__main__":
    data = torch.rand(size=[16,1,120,120])
    model = UNetModel(n_channels=1,bilinear=True)
    res = model(data)

    print(res.shape)
    