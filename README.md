
# Styleformer

<a href="https://colab.research.google.com/drive/1exy4kS-OdsHHA_yY9dzOjQCzAkz--6q6?authuser=4#scrollTo=V5Xado9PNS74" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>
<a href="./docs/slides.pdf" target="_parent"><img src="https://img.shields.io/badge/Slides-PowerPoint-orange" alt="Open In Colab"/></a>

### An implementation of a style-based generator convolution-free based on transformers.


GAN's  *(Generative Adversarial Networks)* models are living a huge success since they were introduced in 2014, GAN's state of the art models uses mostly a convolutional backbone that suffers of locality problem and this led to difficulties to capture global features.
In this work, taking inspiration from the reference paper **[1]** as well strongly based on Stylegan2 implementation for python **[2]**, we replicate and build some computational approaches to implement a strong, but also light, style-based generator with a convolution-free structure.

GAN's methods, and also our implementation of Styleformer (thinked to reduce the computational cost) are very demanding on GPUs, specially during the training phase, which means that you have to own a very high-end GPU (preferably more than one).
Google Colab is a free service that is suited to let anybody to write and execute arbitrary python code through the browser, and is especially well suited to machine learning, data analysis and education. Colab fits very well with Stylegan and this is the main reason about our choice to enanche Stylegan experience through it.
Anyways, in some case it's useful to have a local environment up and running, so we will also see how to setup everything to work with Windows in a wsl environment.

## Data & methodology
[Dataset + Architecture]

## Repository files
- network_Generator => ...
- network_Discriminator => ...
- ...

## How to run locally through WSL

[Steps]

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
