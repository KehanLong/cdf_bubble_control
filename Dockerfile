FROM nvidia/cuda:11.8.0-base-ubuntu22.04

# Install system dependencies
RUN apt update && apt install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    python3-pip \
    python3-dev \
    wget \
    libboost-all-dev \
    libeigen3-dev \
    libode-dev \
    lsb-release \
    && rm -rf /var/lib/apt/lists/*

# Install Python packages
RUN pip3 install --no-cache-dir \
    numpy==1.26.4 \
    torch==2.2.1 \
    torchvision==0.17.1 \
    torchaudio==2.2.1 \
    matplotlib==3.5.1 \
    trimesh==4.5.1 \
    cvxpy==1.6.0 \
    casadi==3.6.7 \
    imageio==2.35.1 \
    pybullet==3.2.6 \
    PyMCubes==0.1.6 \
    "pyglet<2" \
    scipy==1.14.1 \
    scikit-learn==1.5.2 \
    python-igraph==0.10.8 \
    --extra-index-url https://download.pytorch.org/whl/cu118

# Install OMPL dependencies and build tools
RUN apt update && apt install -y --no-install-recommends \
    wget \
    libboost-all-dev \
    libeigen3-dev \
    libode-dev \
    && rm -rf /var/lib/apt/lists/*

# Download and run OMPL installation script
RUN wget https://ompl.kavrakilab.org/core/install-ompl-ubuntu.sh && \
    chmod u+x install-ompl-ubuntu.sh && \
    ./install-ompl-ubuntu.sh --python && \
    rm install-ompl-ubuntu.sh

# Set working directory
WORKDIR /workspace

# Set Python buffering to unbuffered
ENV PYTHONUNBUFFERED=1