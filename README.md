Neural Configuration-Space Barriers for Manipulation Planning and Control [[Paper]](https://arxiv.org/abs/2409.13865)
===========================================


This repository contains the official implementation for the paper "Neural Configuration-Space Barriers for Manipulation Planning and Control".

If you find our work useful, please consider citing our paper:
```

``` 

## 🚀 Quick Start
Clone the repository: 
```bash
git clone <repository_url>
cd <repository_name>
```

## 📦 Dependencies
This code has been tested on Ubuntu 22.04 LTS. You can run it using either Docker (recommended) or conda environment.

### 🐳 Using Docker (Recommended)
1. Build the Docker image:
```bash
docker build -t bubble_cdf_planner:latest .
```

2. Make the run script executable:
```bash
chmod +x run_docker.sh
```

3. Run the container:
```bash
./run_docker.sh
```

This will start an interactive shell in the container. You can then run the examples as described below.

### 🐍 Using Conda (Alternative)
If you prefer using conda, you can set up the environment:
```bash
conda create -n environment.yaml
conda activate arm_cdf_planning
```

You can install miniconda from [here](https://docs.conda.io/en/latest/miniconda.html).

## 2D Examples
Run the file: 
```bash
python 2Dexamples/main_planning.py
```

#### 🧠 Neural Truncated CDF Training
*   Default training dataset is saved in training_data/

To train the neural T-CDF, run the file:
```bash
python main_cdf.py
```
The training parameters can be adjusted in training/config_3D.py. 



#### 🤖 Safe Bubble Planning




