Neural Configuration-Space Barriers for Manipulation Planning and Control [[Paper]](https://arxiv.org/)
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

You will additionally need to install OMPL depedencies: https://ompl.kavrakilab.org/installation.html

## 2D Examples

### Neural CDF Training

Default training dataset is saved in `2Dexamples/cdf_training/data/`. To train the neural CDF:
```bash
python 2Dexamples/cdf_training/cdf_train.py
```

The default trained model is saved in `2Dexamples/trained_models/`.

### Bubble-CDF Planning

To run the bubble-CDF planning, run the following command:
```bash
python 2Dexamples/main_planning.py
```

Baseline Comparison:
To compare with baselines using OMPL's planners, run the following command:
```bash
python 2Dexamples/planning_benchmark.py
```

### DRO-CBF Control

To run the DRO-CBF control, run the following command:
```bash
python 2Dexamples/main_control.py
```









