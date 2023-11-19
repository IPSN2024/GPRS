#encoding=utf-8

import torch
from torch.utils.data import Dataset,DataLoader
import numpy as np


class RTIDataSet(Dataset):
    def __init__(self,fileName) -> None:
        self.filenamelist = []
        self.groundList = []
        #self.labelList = []
        with open(fileName,"r",encoding="utf-8") as f:
            flines = f.readlines()
            for line in flines:
                line = line.strip().split("\t")
                self.filenamelist.append(line[0])
                self.groundList.append(line[1])
                #self.labelList.append(line[2])
    def __getitem__(self, index):
        return torch.load(self.filenamelist[index]).float(),torch.from_numpy(np.load(self.groundList[index])).float()
    def __len__(self):
        return len(self.filenamelist)
    

if __name__ == "__main__":
    dataset = RTIDataSet("../Record/TrainSampleCNN.txt")
    dataloader = DataLoader(dataset=dataset,batch_size=512,shuffle=False)
    for idx,(traindata,trainlabel) in enumerate(dataloader):
        print(idx,traindata.shape,trainlabel.shape)