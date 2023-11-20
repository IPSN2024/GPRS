#encoding=utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim 
from dataloader import RTIDataSet
from torch.utils.data import DataLoader, Dataset
from network60 import RadioTomoNet
from UNet import UNetModel
from evaluate import LoadFileForEva,ImageDataNormalize
import os


testDataset = RTIDataSet("../Datafile/Test.txt")
testloader = DataLoader(dataset=testDataset,batch_size=128,shuffle=False,num_workers=4,pin_memory=True)

if __name__ == "__main__":
    Radio = torch.load("../PretrainedModel/Radio.pth",map_location="cpu")
    UNet = torch.load("../PretrainedModel/UNet.pth",map_location="cpu")
    if not os.path.exists("../TestResult/"):
        os.mkdir("../TestResult/")  
    ssim_sum_u = 0.0
    psnr_sum_u = 0.0
    uqi_sum_u = 0.0
    count = 0
    with torch.no_grad():
        Radio.eval()
        UNet.eval()
        for idx,(testdata,testlabel) in enumerate(testloader):
            print(idx)
            #testdata = testdata.cuda()
            #testlabel = testlabel.cuda()
            res = Radio(testdata)
            res = res.unsqueeze(1)
            res = UNet(res)
            res = res.squeeze(1)
            torch.save(res,"../TestResult/"+str(idx))
	
            batchSize = testdata.shape[0]
            for kk in range(batchSize):
                imgip = res[kk].detach().cpu().numpy()
                imgt = testlabel[kk].detach().cpu().numpy()
                ssimvalueu,psnrvaleu,uqivalueu = LoadFileForEva(imgt,imgip)
                ssim_sum_u += ssimvalueu
                psnr_sum_u += psnrvaleu
                uqi_sum_u += uqivalueu
                count += 1
        ssim_avg_u = ssim_sum_u / count
        psnr_avg_u = psnr_sum_u / count
        uqi_avg_u = uqi_sum_u / count
        with open("../TestResult/Evaluation.txt","a+",encoding="utf-8") as fe:
            fe.write("UNet:")
            fe.write("\t")
            fe.write("SSIM:")
            fe.write("\t")
            fe.write(str(ssim_avg_u))
            fe.write("\t")
            fe.write("PSNR:")
            fe.write("\t")
            fe.write(str(psnr_avg_u))
            fe.write("\t")
            fe.write("UQI:")
            fe.write("\t")
            fe.write(str(uqi_avg_u))
            fe.write("\n")