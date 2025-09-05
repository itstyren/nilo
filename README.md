# Submission

## Installation Guide

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

## Running Guide

To run the project, use the following command from the root directory:

```bash
bash grg/scripts/train_scripts/local_md.sh
```

This will start the training process using the provided local script.

---

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

These scripts will train and visualize the coin game environment respectively.

