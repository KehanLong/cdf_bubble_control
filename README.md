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

If using conda environment, you will additionally need to install OMPL depedencies: https://ompl.kavrakilab.org/installation.html

## 2D 2-link Planar Robot

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

## PyBullet (xArm6)

### Neural SDF/CDF Training

The default trained SDF/CDF models are saved in `xarm_pybullet/trained_models/`.

### Bubble-CDF Planning
To run the bubble-CDF planning in PyBullet:
```bash
# Default settings
python xarm_pybullet/xarm_planning.py

# Custom settings
python xarm_pybullet/xarm_planning.py --goal [0.7,0.1,0.6] --planner bubble --seed 42 --gui True --early_termination True

# Available planners: bubble, bubble_connect, sdf_rrt, cdf_rrt, lazy_rrt, rrt_connect ...
# early termination: Stop after first valid path or explore all goal configurations
```


### DRO-CBF Control
To run the DRO-CBF control:
```bash
# Default settings (with dynamic obstacles)
python xarm_pybullet/xarm_control.py

# Custom settings
python xarm_pybullet/xarm_control.py --goal [0.7,0.1,0.6] --planner bubble --controller clf_dro_cbf --dynamic True --gui True --early_termination True

# Available options:
# - planners: bubble, sdf_rrt, cdf_rrt, rrt_connect, lazy_rrt ...
# - dynamic: Whether to use dynamic obstacles
# - controllers: pd, clf_cbf, clf_dro_cbf
```










