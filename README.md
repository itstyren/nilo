# Submission

## 1. System Requirements

### Software Dependencies
- Operating System: Linux (Ubuntu 20.04/22.04), macOS, or Windows with WSL2
- Python 3.11
- CUDA 11.8 (for GPU support)

### Tested Versions
- Ubuntu 22.04 LTS
- Python 3.11
- PyTorch 2.x with CUDA 11.8
- NVIDIA Driver 520+

### Hardware Requirements
- Minimum: 8GB RAM, 4 CPU cores
- Recommended: 16GB+ RAM, NVIDIA GPU with 8GB+ VRAM, 8+ CPU cores
- Tested on: Tesla V100

---

## 2. Installation Guide

Follow the steps below to set up the environment and install the required dependencies.

First, create a new Conda environment and install PyTorch:
```bash
conda create -n grg python=3.11
conda activate grg
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
```

Then, install this project by returning to the root directory and running:
```bash
cd stable_baselines3/ && pip install -e .
cd ../ && pip install -e .
```

### Installation Time
Typical install time: 10-15 minutes on a standard desktop computer with stable internet connection.

---

## 3. Demo

### Instructions to Run Demo
To run a quick demonstration, use the following command from the root directory:
```bash
bash grg/scripts/train_scripts/local_coin.sh
```

### Expected Output
The training script will produce:
- Training progress logs with episode rewards
- Checkpoint saves in the logs directory
- Performance metrics including average reward and loss values

### Expected Run Time
Demo run time: Approximately 5-10 minutes on a standard desktop computer with GPU.

---

## 4. Instructions for Use

### Running Training Scripts

To run the main training process, use the following command from the root directory:
```bash
bash grg/scripts/train_scripts/local_md.sh
```

This will start the training process using the provided local script.

### Run Coin Game
To train on the coin game environment, use:
```bash
bash grg/scripts/train_scripts/local_coin.sh
```

### Render Coin Game
To render the coin game, use:
```bash
bash grg/scripts/render/render_coin.sh
```

These scripts will train and visualise the coin game environment respectively.

---

## 5. Reproduction Instructions

To reproduce results from the paper:
1. Follow the installation guide above
2. Run the training scripts for each environment
3. Results and checkpoints will be saved in the logs directory

Full training time: 6-48 hours per environment depending on hardware configuration.


