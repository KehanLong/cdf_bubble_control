Safe Bubble Planning in Truncated Configuration Field [[Paper]](https://arxiv.org/abs/2409.13865)
===========================================


This repository contains the official implementation for the paper "Safe Bubble Planning with Neural Truncated Configuration Field".

If you find our work useful, please consider citing our paper:
```

``` 


## 🚀 Quick Start
Clone the repository: 

```
git clone 
cd 
```

## 📦 Dependencies
This code has been tested on Ubuntu 22.04 LTS. To set up the environment:

```
conda create -n environment.yaml
conda activate arm_cdf_planning
```

You can install miniconda from [here](https://docs.conda.io/en/latest/miniconda.html). 

## 2D Examples
Run the file: 
```
2Dexamples/main_planning.py
```


#### 🧠 Neural Truncated CDF Training

*   Default training dataset is saved in training_data/


To train the neural T-CDF, run the file:
```
main_cdf.py
```
The training parameters can be adjusted in training/config_3D.py. 



#### 🤖 Safe Bubble Planning




