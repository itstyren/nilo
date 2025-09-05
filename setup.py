from setuptools import setup, find_packages
import os

setup(
    name="grg",
    version="1.0.0",
    description="Parent project containing sub-packages",
    author="Your Name",
    author_email="your_email@example.com",
    packages=find_packages(),  # Automatically finds sub-packages 
    install_requires=[
        # List dependencies shared across the project
        "numpy",
        "pandas",
        'wandb',
        "supersuit",
        "tensorboard",
        "moviepy",
        "torchinfo",
        "tensorboardX",
        
    ],
    extras_require={
        "stable_baselines3": [],
        "dev": [],
    },
)
