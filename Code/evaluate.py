#encoding=utf-8

import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

from torch.utils.data import DataLoader
from dataloader import RTIDataSet
import os
import natsort

def SSIMScore(TruthLabel,PredictImg):
    '''
    #用for循环对每张图像进行计算SSIM结果
    PredictImg: Height,Width
    TruthLabel: Height,Width
    '''
    return ssim(PredictImg,TruthLabel)

def PSNRScore(TruthLabel,PredictImg):
    '''
    #用for循环对每张图像进行计算PSNR结果
    PredictImg: Height,Width
    TruthLabel: Height,Width
    '''
    return psnr(TruthLabel,PredictImg)
def ImageDataNormalize(imgedata):
    maxvalue = np.max(imgedata)
    minvalue = np.min(imgedata)
    imgedata = (imgedata - minvalue) / (maxvalue - minvalue)
    return imgedata


class ImageEvalue(object):
    def image_mean(self, image):
        mean = np.mean(image)
        return mean
    def image_var(self, image, mean):
        m, n = np.shape(image)
        var = np.sqrt(np.sum((image - mean) ** 2) / (m * n - 1))
        return var
    def images_cov(self, image1, image2, mean1, mean2):
        m, n = np.shape(image1)
        cov = np.sum((image1 - mean1) * (image2 - mean2)) / (m * n - 1)
        return cov
    def UQI(self, O, F):
        '''
        :param O: 原始图像
        :param F: 滤波后的图像
        '''
        meanO = self.image_mean(O)
        meanF = self.image_mean(F)
        varO = self.image_var(O, meanO)
        varF = self.image_var(F, meanF)
        covOF = self.images_cov(O, F, meanO, meanF)
        UQI = 4 * meanO * meanF * covOF / ((meanO ** 2 + meanF ** 2) * (varO ** 2 + varF ** 2))
        return UQI

evaob = ImageEvalue()
def LoadFileForEva(TrueImage,PredictImage):
    return SSIMScore(TrueImage,PredictImage), PSNRScore(TrueImage,PredictImage), evaob.UQI(TrueImage,PredictImage)

    
   