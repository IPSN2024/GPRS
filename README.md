# GPRS
We introduce GPRS, a data-driven model designed for below-ground biomass sensing. GPRS is capable of sensing the blow-ground biomass, including sizes, shapes and positions, through radio tomographic imaging. We extensively evaluate GPRS's sensing capabilities and validate its effectiveness in multiple scenarios, including multiple underground potato tubers, random positions, different soils with different moisture, leave-k-out potato tubers and domain adaptation(crossing soils and crossing environments). 

We have released several pre-trained models and part of the corresponding data files in releases, which can be used for performance testing. 

Some visualization results of GPRS are shown as follows. For demonstration purposes, we provide the imaging results in the dual-potato tubers scenario, the random positions scenario, and the leave-k-out potato tubers scenario, which are representative of underground biomass sensing. Please note that the pixel size of all generated results is 1cm, and the monitored size is configured as 60cm $\times$ 60cm. 

|               | Predicted Result|Ground Truth|
| ------------- | -------------| -------------   |
|Dual-potato tubers|<img src="Img/double_3.png">  <img src="Img/double_4.png">|<img src="Img/double_3_g.png">  <img src="Img/double_4_g.png">|
|Random positions|<img src="Img/rotate_0.png">  <img src="Img/rotate_0_n.png">|<img src="Img/rotate_1.png">  <img src="Img/rotate_1_n.png">|
|Leave-2-out| | |
|Leave-4-out| | |


An intuitive comparison between GPRS and three other prevalent models is shown as follows. For demonstration purposes, we provide 
