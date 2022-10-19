




# Styleformer
### Implementation of a style image generator, convolution-free and based on Transformer model.
<a href="https://colab.research.google.com/drive/1exy4kS-OdsHHA_yY9dzOjQCzAkz--6q6?authuser=4" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<a href="./docs/slides.pdf" target="_parent"><img src="https://img.shields.io/badge/Slides-PowerPoint-orange" alt="Open In Colab"/></a>


GAN's  *(Generative Adversarial Networks)* models are living a huge success since they were introduced in 2014, GAN's state of the art models uses mostly a convolutional backbone that suffers of locality problem and this led to difficulties to capture global features.
In this work, taking inspiration from the reference paper **[1]** as well strongly based on Stylegan2 implementation for python **[2]**, we replicate and build some computational approaches to implement a strong, but also light, style-based generator with a convolution-free structure.

GAN's methods, and also our implementation of Styleformer (thinked to reduce the computational cost) are very demanding on GPUs, specially during the training phase, which means that you have to own a very high-end GPU (preferably more than one).
Google Colab is a free service that is suited to let anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education. Colab fits very well with Stylegan and this is the main reason about our choice to enhance Styleformer experience through it.
Anyways, in some case it's useful to have a local environment up and running, so we will also see how to setup everything to work with Windows in a wsl environment.

## Architecture

**Applying a NPL strategies to image generation**

As mentioned before, Styleformer is based on a NPL-native technique which is the Transformer **[3]**. Transformer is a simple network architecture based on the attention mechanism that recently, since it's release, it has become an integral part of applying deep learning. Attention is meant to mimic the cognitive attention, in this sense its responsability is to put the focus on small but significative details of an image, a token or any other significative data. The Transformer can be described as *"The first transduction model relying entirely on self-attention to compute representations of its input and output without using sequence-aligned RNNs or convolution"* **[3]**. Where transduction means conversion of input sequences into output sequence and the whole idea is to handle the dependencies between this two poles only with attention. 
Due to the great performance obtained by the Transformer model with attention and self-attention, recently reserchers among the globe are trying to replace themost known convolutional operation in GAN with the Transformer model to increase the generation quality and surprisingly obtaining comparable performances with the state of the art GAN's models. 

Styleformer tries to apply Transformer to Stylegan2-ADA **[2]**, with the transformer only generator we can solve some of the most known issues of using convolution network, resulting in a better handling of long-range dependency and better understanding of global feature thanks to self-attention, furthermore overcoming locality problems. 
However, Transformer presents also some drawbacks, indeed, for higher dimension data it results prehibitevely costly since the self attention-mechanism have a cost of *O(n^2)*, for this reason we have used Linformer **[4]** to overcome this issuse. Linformer reduce the complexity from quadratic to linear by projecting key and value to k dimension while applying self-attention, so that the cost results to be O(nk) with k defined as the projection dimension for key and value.

**How Styleformer works?**

<img src="/docs/images/architecture.jpg" alt="Styleformer architecture" width="400"/>

Styleformer generator is conditioned on a learnable constant input and combined with a learnable positional encoding (as seen in the Transformer model **[3]**) which is a scheme throguh which the knowledge about the order of a input is mantained.
The constant input (8x8) is flattened (64) to enter the Transformer-based encoder, then the input passes through the Styleformer encoder. Each resolution passes through several encoder blocks and eventually we proceed with a bilinear upsample operation by reshaping encoder output to the form of square feature map. After upsampling, flatten process is carried out again to match the input form of the Styleformer encoder, followed by adding positional encoding in the form of a learned parameter. This process will repeat until the feature map resolution reaches the target image resolution.
For each resolution, the number of the Styleformer encoder and hidden dimension size can be chosen as hyperparameters, each for these parameters, can change for each resolution.

Let's see in the details what happen in the encoder blocks:

<img src="/docs/images/encoder.jpg" alt="Modified enchoder architecture" width="400"/>

StyleGAN generates an image by receiving different inputs in the form of style vectors, and similar to it, we need different style vector for each module, therefore for each operation, to learn effectively.
As said before, we need a Transformer-based generator that generates images applying a style vector, for this reason we need a style modulation and demodulation methods, in which we apply style vector to achieve the self-attention operation.
Attention mechanism can be seen as built in two steps, a *preparation* module in which we compute **Q**uery, **K**ey and **V**alue and a *main module* in which the attention operation is applied.

This implementations, contrarily to the original structure of Transformer, uses a pre-Layer Normalization procedure.  Indeed, the Layer Normalization of the existing Transformer comes after a linear layer that integrates multi-heads which, in our case, could led some disturbs in generating the attention map (style modulation would be applied before entering the prepare module of the next encoder). So, given that the role of layer Normalization in a Transformation is the preparation of generating an attention map, we need to apply normalization before the prepare module of the Attention, solving potential issues with the learning process.

In the *prepare module,* after the normalization, we have the demodulation for Query, Key and Value. Styleformer encoder creates **Q**, **K** and **V** through linear operation to the input feature map scaled with Style Input Vector. After these operations, V will be modulated again but this time with a different vector called *"Style Value"*, thus a demodulation to remove the scaled effect of Style Input is required. This double style injection led us to obtain different style vector for each step! It's important to mention that we also need a demodulation of Q and K before creating the attention map, since the dot product to create it could become very expansive otherwise. 

After demodulation of Q, K and V, we can find the core of Self-Attention, here we have the production of the attention map and the weighted sum of V with attention map itself *(increased multi-head Self-Attention)*, and then performs linear operation. The Self-Attention mechanism is needed to avoid to use always the same kernel for each section (pixel and channel), indeed otherwise we would have only one huge kernel for each channel, and so diversity in generated image would be decreased.
As mentioned, the pixel-communication(self-attention) and channel-communication(multi-head integration) in the Transformer encoder is separated. We overcome the problem of using a huge kernel by increasing the number of head of multi-head attention, then the created attention map will be different for each head. 

So a attention map will be created for each head, making channels in each head meet different kernels. But if we increase too much the number of heads, it may cause attention map to not be properly created, resulting in poor performance. So, as demonstrated in the paper **[1]**, the increasing of the number of heads improves performance only when the depth is at least 32.

A further demodulation is needed at this point, since a Encoder output will be input for next one. It's important to notice also that we are using ReLu activation function and we add bias and noise at the end of each encoder block. After passing blocks we reshape it to the form of a square feature map (Unflatten) and we proceed with the cited bilinear unsample. Furthermore, each output in each resolution is converted into a RGB channel with a RGB layer inspired from **[2]** implementation.

## Dataset & Evaluation

As said in the introduction, to train with GAN's an high end GPU is needed, more than one actually is better. Due to lack of material resources we could not replicate the same results obtained from **[1]** in a decent amount of time, for completeness and to ensure the validity of the project we are going to present the results obtained with **[1]** compared with our trained model with Colab.

For the metrics to evaluate our implementation we have choosen the **Frechet Inception Distance**, known as FID, which is a method for comparing the statistics of two distributions by computing the distance between them. In GANs, the FID method is used for computing how much the distribution of the Generator looks like the distribution of the Discriminator. The conseguence is that the lower is the FID, the better is the GAN.

**CIFAR-10:** 

this is widely used as a benchmark dataset. They used 50K images(32x32) at the training set, without using label.
With the pre-trained pickle Styleformer records FID 2.82, and IS 9.94, which is comparable with current state-of-the-art. 

With our implementation after 50 minutes of training with Colab, Styleformer recorded FID 95.49, and IS xx.xx.
With Styleformer code from [1] after 50 minutes of training with Colab, recorded FID 86.63 and IS 4.66.
We are aware that the training in this small amount of time is not a complete information, but is significative to show the goodness of the implementation. 
Actually we are working to train the network for an higher amount of time and eventually with more hardware resources to have a more accurate estimation of performances.

## Repository Structure

    ├── calc_metrics.py
    ├── dataset_tool.py                 // Dataset Tools from Stylegan
    ├── dnnlib                          // Light version of dnnlib
        └── util.py
    ├── docs                            // Docs available in our repository
    ├── generate.py                     // Generation function definition
    ├── legacy.py                       // PKL handle utilities
    ├── metrics                         // Folder containing metrics methods
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
Sometimes having a working environment in local is very useful, unfortunately setup Stylegan environment is not a trivial task, expecially on Windows systems.
While the majority of the community uses a Docker based approach, we want to share our guide to setup the whole environment through Windows Subsystems Linux and Anaconda.

The user should [setup WSL2](https://learn.microsoft.com/it-it/windows/wsl/install) and eventually download from Windows store the WSL image for [Ubuntu version 18.04](https://apps.microsoft.com/store/detail/ubuntu-1804-on-windows/9N9TNGVNDL3Q?hl=en-us&gl=us).

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

Furthermore, you can try to generate an image with one of the pretrained pickle (or a new generated one as well):

    // Clone our repository
    $ git clone https://github.com/Jeeseung-Park/Styleformer.git
    $ cd Styleformer
    // Generate a random image using CelebA pickle
    $ python generate.py --outdir=out --seed=100-105 --network=./pre_trained/Pretrained_CelebA.pkl

The output should be like the image above

![Generate some random image](https://i.ibb.co/8jzqLtW/image.png)

## Some samples generated with our Colab
![Colab Image sample](docs/sample_img1.png)

![Colab Video sample](docs/sample_video1.mp4)

![Colab Finetune sample](docs/sample_finetune1.png)

## Authors
* ##### [Fabio Caputo](https://it.linkedin.com/in/fabio-caputo-41163b171)
* ##### [Weihao Peng](https://it.linkedin.com/in/weihao-peng-a872b320a)

## Reference papers

[1] [**Styleformer: Transformer based Generative Adversarial Networks with Style Vector**](https://arxiv.org/abs/2106.07023)

Jeeseung Park, Younggeun Kim.

[2] [**Analyzing and Improving the Image Quality of StyleGAN**](https://arxiv.org/abs/1912.04958)

Tero Karras, Samuli Laine, Miika Aittala, Janne Hellsten, Jaakko Lehtinen, Timo Aila.

[3] [**Attention Is All You Need**](https://arxiv.org/abs/1706.03762)

Ashish Vaswani, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N. Gomez, Lukasz Kaiser, Illia Polosukhin.

[4] [**Linformer: Self-Attention with Linear Complexity**](https://arxiv.org/abs/2006.04768)

Sinong Wang, Belinda Z. Li, Madian Khabsa, Han Fang, Hao Ma
