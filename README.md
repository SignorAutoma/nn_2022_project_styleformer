


# Styleformer
### An implementation of a style based generator convolution-free basend on transformers.
<a href="https://colab.research.google.com/drive/1exy4kS-OdsHHA_yY9dzOjQCzAkz--6q6?authuser=4#scrollTo=V5Xado9PNS74" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<a href="./docs/slides.pdf" target="_parent"><img src="https://img.shields.io/badge/Slides-PowerPoint-orange" alt="Open In Colab"/></a>


GAN's  *(Generative Adversarial Networks)* models are living a huge success since they were introduced in 2014, GAN's state of the art models uses mostly a convolutional backbone that suffers of locality problem and this led to difficulties to capture global features.
In this work, taking inspiration from the reference paper **[1]** as well strongly based on Stylegan2 implementation for python **[2]**, we replicate and build some computational approaches to implement a strong, but also light, style-based generator with a convolution-free structure.

GAN's methods, and also our implementation of Styleformer (thinked to reduce the computational cost) are very demanding on GPUs, specially during the training phase, which means that you have to own a very high-end GPU (preferably more than one).
Google Colab is a free service that is suited to let anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education. Colab fits very well with Stylegan and this is the main reason about our choice to enanche Stylegan experience through it.
Anyways, in some case it's useful to have a local environment up and running, so we will also see how to setup everything to work with Windows in a wsl environment.

## Data & methodology
[Dataset + Architecture]

## Repository Structure

    ├── calc_metrics.py
    ├── dataset_tool.py                 // Dataset Tools from Stylegan
    ├── dnnlib                          // Light version of dnnlib
        └── util.py
    ├── docs                            // Docs available in our repository
    ├── generate.py                     // Generation function definition
    ├── legacy.py                       // PKL handle utilities
    ├── metrics                         // Folder containing metrics methods
    ├── pre_trained                     // Pre-trained samples from [1]
        ├── Pretrained_CIFAR10.pkl
        ├── Pretrained_CelebA.pkl
        └── Pretrained_LSUNchurch.pkl
    ├── torch_utils                     // Torch utilities
    ├── train.py                        // Training main methods
    └── training                        // Networks utilities
        ├── augment.py
        ├── dataset.py
        ├── loss.py
        ├── networks.py                 // Legacy stylegan2 network implementation
        ├── networks_Discriminator.py   // Legacy discrimination stucture
        ├── networks_Generator.py       // New generation stucture
        └── training_loop.py			

## How to run locally through WSL
#### Styleformer ~ Win 11 and wsl Ubuntu 18.04
Setup Stylegan environment is not a trivial task, expecially on Windows systems, while the majority of the community uses a Docker based approach we want to share our guide to setup the whole environment through Windows Subsystems Linux and Anaconda.

The user should [setup WSL2](https://learn.microsoft.com/it-it/windows/wsl/install) and eventually download from Windows store [Ubuntu version 18.04](https://apps.microsoft.com/store/detail/ubuntu-1804-on-windows/9N9TNGVNDL3Q?hl=en-us&gl=us).

From wsl install Anaconda

    $ wsl
    $ wget https://repo.anaconda.com/archive/Anaconda3-2020.11-Linux-x86_64.sh
    $ bash Anaconda3-2020.11-Linux-x86_64.sh

Create a conda environment with python 3.7

    $ conda create --name pytorch python=3.7
    $ conda activate pytorch

Download Cuda Toolkit for ubuntu and Wsl driver

    $ wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-wsl-ubuntu.pin
    
    $ sudo mv cuda-wsl-ubuntu.pin /etc/apt/preferences.d/cuda-repository-pin-600
    
    $ wget https://developer.download.nvidia.com/compute/cuda/11.1.1/local_installers/cuda-repo-wsl-ubuntu-11-1-local_11.1.1-1_amd64.deb
    
    $ sudo dpkg -i cuda-repo-wsl-ubuntu-11-1-local_11.1.1-1_amd64.deb
    
    $ sudo apt-key add /var/cuda-repo-wsl-ubuntu-11-1-local/7fa2af80.pub
    
    $ sudo apt-get update
    
    $ sudo apt-get -y install cuda

Install pre-requisites for Styleformer and torch

    pip install click requests tqdm pyspng ninja imageio-ffmpeg==0.4.3
    
    pip3 install torch==1.8.2 torchvision==0.9.2 torchaudio===0.8.2 --extra-index-url https://download.pytorch.org/whl/lts/1.8/cu111

Finally check if the environment is working, so activate conda environment from wsl and try to import torch and get its version as in the following snippet:

![Check environment](https://i.ibb.co/zXVq6Sv/image.png)

Furthermore, you can try to generate an image with one of the pretrained pickle:

    // Clone our repository
    $ git clone https://github.com/Jeeseung-Park/Styleformer.git
    $ cd Styleformer
    // Generate a random image using CelebA pickle
    $ python generate.py --outdir=out --seed=100-105 --network=./pre_trained/Pretrained_CelebA.pkl

The output should be like the image above

![Generate some random image](https://i.ibb.co/8jzqLtW/image.png)

---
## Authors
* ##### [Fabio Caputo](https://it.linkedin.com/in/fabio-caputo-41163b171)
* ##### [Weihao Peng](https://it.linkedin.com/in/weihao-peng-a872b320a)
---
## Reference papers

[1] **Styleformer: Transformer based Generative Adversarial Networks with Style Vector**  
Jeeseung Park, Younggeun Kim

[2] **Analyzing and Improving the Image Quality of StyleGAN**  
Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila
