# GPRS
We introduce GPRS, a data-driven model designed for below-ground biomass sensing. GPRS is capable of sensing the blow-ground biomass, including sizes, shapes and positions, through radio tomographic imaging. We extensively evaluate GPRS's sensing capabilities and validate its effectiveness in multiple scenarios, including multiple underground potato tubers, random positions, different soils with different moisture, leave-k-out potato tubers and domain adaptation(crossing soils and crossing environments). 

We have released the pre-trained model and part of the corresponding data files in [releases](https://github.com/IPSN2024/GPRS/releases/tag/v1.0.0.0), which can be used for performance testing. 

## 1. Visualization Results
Some visualization results of GPRS are shown as follows. For demonstration purposes, we provide the imaging results in the dual-potato tubers scenario, the random positions scenario, and the leave-k-out potato tubers scenario, which are representative of underground biomass sensing. Please note that the pixel size of all generated results is 1cm, and the monitored size is configured as 60cm $\times$ 60cm. 

|               | Predicted Result|Ground Truth|
| ------------- | -------------| -------------   |
|Dual-potato tubers|<div align=center><img src="Img/double_3.png" width="50%"></div>  <div align=center><img src="Img/double_4.png" width="50%"></div>|<div align=center><img src="Img/double_3_g.png" width="50%"></div>  <div align=center><img src="Img/double_4_g.png" width="50%"></div>|
|Random positions|<div align=center><img src="Img/rotate_0.png" width="50%"></div>  <div align=center><img src="Img/rotate_1.png" width="50%"></div>|<div align=center><img src="Img/rotate_0_n.png" width="50%"></div>  <div align=center><img src="Img/rotate_1_n.png" width="50%"></div>|

As shown, GPRS generates images that accurately demonstrate sizes, shapes and positions. 

## Prerequisite and Evaluation
### 2.1 Prerequisite
GPRS is implemented with Python 3.8 and PyTorch 1.9.1. We manage the development environment using Conda(version: 4.10.3). Execute the following commands to configure the development environment. Note that, our pre-trained model and testing dataset for imaging two potato tubers has been provided in [releases](https://github.com/IPSN2024/GPRS/releases/tag/v1.0.0.0). You can directly download them for evaluation. 

- Create a conda environment called GPRS based on Python 3.8, and activate the environment.
     ```python
    conda create -n GPRS python=3.8
    conda activate GPRS 
    ```
- Install PyTorch, as well as other required packages,
  ```python
    pip3 install torch
    pip3 install numpy
    pip3 install scikit-learn
    pip3 install scikit-image
  ```
- Download or git clone the GPRS project from [releases](https://github.com/IPSN2024/GPRS/releases/tag/v1.0.0.0). Download and unzip PretrainedModel.zip, 20231021Samples.zip, and Ground1021.zip in releases to the project directory.
  ```python
     unzip PretrainedModel.zip -d [GPRS root directory]
     unzip 20231021Samples.zip -d [GPRS root directory]
     unzip Ground1021.zip -d [GPRS root directory]
  ```
- Perform the testing process by running Code/Test.py
```python
     python3 Code/Test.py
```
- The predicted results can be visualized by running Code/visualization.py. Upon executing Code/Test.py, please be advised that the anticipated outcomes and assessment metrics will be archived in the "TestResult" directory. For visualization purposes, feel free to select any file, excluding those with a ".txt" extension. 

### 2.2 Performance on the unseen potato scenario

The ratio of training potato tubers to testing ones is configured as 8:1, 2:7, 6:3, and 5:4. Meanwhile, we conduct the experiment in two monitored areas with different sizes, including 40cm $\times$ 40cm and 60cm $\times$ 60cm.

<div align=center><img src="Results/Unseen.png" width="90%"></div>

It is evident that our method consistently attains SSIM and UQI scores exceeding 0.95, along with PSNR scores surpassing 28.31 in the monitored area measuring 60cm $\times$ 60cm.

### 2.3 Evaluation on Soil Moisture Levels

#### 2.3.1 Performance on Various Soils. 

We collect RSS measurements from two distinct soil types characterized by varying levels of soil moisture: 7.10% and 11.20%. In each monitored area with the specified monitored size, the pixel size of the monitored area is fixed at 1𝑐𝑚.

<div align=center><img src="Results/Different_Soils.png" width="30%"></div>

Our approach attains SSIM scores surpassing 0.99, PSNR surpassing 37.13, and UQI surpassing 0.95 for soils with moisture levels of 7.10% and 11.20% across all monitored sizes of
40cm $\times$ 40cm, 60cm $\times$ 60cm and 100cm $\times$ 100cm.

#### 2.3.2 Performance on Domain Adaptation

The following figures demonstrate the results of the domain adaptation experiment across different soils with different moistures. The x-axis denotes the number of samples employed from the target domain for fine-tuning the pre-trained model. We incrementally increase the number of samples utilized in the fine-tuning process of the pre-trained model. 

<img src="Results/PSNRSoil.png" width="30%">   &emsp;   <img src="Results/SSIMSoil.png" width="30%">   &emsp; <img src="Results/ComSoil.png" width="30%" height="200%">

When directly applying the pre-trained model to evaluate RSS data collected from the target domain, the SSIM values for the monitored areas sized at 40cm $\times$ 40cm and 60cm $\times$ 60cm are 0.58 and 0.86, respectively. The PSNR values for the same monitored are 21.70 and 18.64. However, when utilizing a random subset of just 6 samples, a significant performance enhancement is observed, leading to an SSIM increase of 0.33 in the monitored area with the size of 40cm $\times$ 40cm. and a PSNR increase of 11.32 in the monitored area with a size of 60cm $\times$ 60cm. The results compared with the baseline method are shown in the last figure, showing improvements in SSIM by 10.06% and PSNR by 36.55% in the monitored size at 60cm $\times$ 60cm.

### 2.4 Evaluation on Different Environments
The following figures show the results of the domain adaptation experiment across different environments. The x-axis denotes the number of samples employed from the target domain for fine-tuning the pre-trained model. We incrementally increase the number of samples utilized in the fine-tuning process of the pre-trained model. 

<img src="Results/PSNRSoilEn.png" width="30%">   &emsp;  <img src="Results/SSIMSoilEn.png" width="30%"> &emsp;  <img src="Results/ComEn.png" width="30%" height="200%">

When employing the pre-trained model directly, SSIM values for the monitored areas of size 40cm $\times$ 40cm and 60cm $\times$ 60cm are 0.46 and 0.84, respectively, with PSNR values of 19.61 and 17.62. With only six samples used for fine-tuning, the SSIM values for these areas increase to 0.88 and 0.94, while the PSNR values rise to 27.63 and 29.90.  Furthermore, when using 18 samples in the fine-tuning process, the SSIM values for these areas significantly increase to 0.99 and 0.98, while the PSNR values improve to 32.24 and 33.96. The results compared with the baseline method are shown in the last figure,  revealing a 12.19% improvement in SSIM for the monitored area sized at 60cm $\times$ 60cm.


  







