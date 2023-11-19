#encoding=utf-8

import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import numpy as np
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

def ShowResSample(fileName):
    res = torch.load(fileName,map_location=torch.device('cpu'))
    resnp = res.detach()
    for i in range(0,10):
        imgedata = resnp[i]
        imgedata = nn.ReLU()(imgedata)
        imgedata = imgedata.numpy()
        plt.imshow(imgedata.T,cmap=plt.cm.gray,origin='lower',interpolation='nearest')
        ax = plt.gca()
        ax.xaxis.set_ticks_position('top')
        ax.invert_yaxis()
        plt.xticks(fontsize=14)
        plt.yticks(fontsize=14)
        plt.show()
        
if __name__ == "__main__":
    ShowResSample("../TestResult/0")