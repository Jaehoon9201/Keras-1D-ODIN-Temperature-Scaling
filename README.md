# Keras-1D-ODIN-Temperature-Scaling
* **Original Idea Paper([Link](https://arxiv.org/abs/1706.02690))** : “Enhancing The Reliability of Out-of-distribution Image Detection in Neural Networks”, 2018 ICLR

* **Reference Keras Code([Link](https://github.com/RRoundTable/ODIN_keras_version) by Wontak Ryu)**

* **Related Paper(my paper)** : “Fault Diagnosis of Inverter Current Sensor Using Artificial Neural Network Considering Out-of-distribution”, 2021 IEEE Energy Conversion Congress and Exposition-Asia(ECCE-Asia), Singapore, May 24-27, 2021   (Accepted, 1st author, To be published)

## Training_with_iteration.py
![image](https://user-images.githubusercontent.com/71545160/118096836-db7de400-b40c-11eb-8f59-9be360c9baa4.png)

![image](https://user-images.githubusercontent.com/71545160/118096818-d456d600-b40c-11eb-9eaa-3bb6289c0cc4.png)

## Classification_Confusion_Matrix.py
![image](https://user-images.githubusercontent.com/71545160/118097809-1df3f080-b40e-11eb-9135-373118c9a52e.png)

![image](https://user-images.githubusercontent.com/71545160/118096889-ec2e5a00-b40c-11eb-8aa7-561552686f2a.png)

## AUROC_without_ODIN_temp_scaling.py
![image](https://user-images.githubusercontent.com/71545160/118096918-f9e3df80-b40c-11eb-8670-3f5c25ef432a.png)

## AUROC_with_ODIN_temp_scaling.py
![image](https://user-images.githubusercontent.com/71545160/118096969-0700ce80-b40d-11eb-859b-3c1c9568bc5e.png)

## How to determine a Temperature scale?
Below code is a part of this repository code.

By this iteration code, i checked which value has the most significant effect to the AUROC.


```python

f3 = open("./T_scaling_results/T_scaling_results.txt", 'w')

#var = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100, 1200, 1300, 1400, 1500, 1600, 1700, 1800, 1900, 2000, 2100, 2200, 2300, 2400, 2500, 2600, 2700, 2800, 2900, 3000, 3100, 3200, 3300, 3400, 3500, 3600, 3700, 3800, 3900, 4000, 4100, 4200, 4300, 4400, 4500,4600, 4700,4800,4900,5000,5100,5200,5300,5400,5500,5600,5700,5800,5900,6000,6100,6200,6300,6400,6500,6600,6700,6800,6900,7000,7100,7200,7300,7400,7500,7600,7700,7800,7900,8000,8100,8200,8300,8400,8500,8600,8700,8800,8900,9000,9100,9200,9300,9400,9500,9600,9700,9800,9900,10000]
#var = [4200]
var = [7900]

for j in range(len(var)):
    temper = var[j]
```

The results of this iteration is represented in the below graph.

![image](https://user-images.githubusercontent.com/71545160/118598144-98df5180-b7e8-11eb-8480-5d97c50fdaaa.png)
