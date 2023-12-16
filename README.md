# Torch Classification

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT) [![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-orange.svg)](https://pytorch.org/) [![CIFAR-100](https://img.shields.io/badge/Dataset-CIFAR--100-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

Torch Classification is a PyTorch-based image classification project that includes the implementation of a convolutional neural network (CNN) for classifying images. The project demonstrates training the model from scratch and utilizing transfer learning with pre-trained weights on the CIFAR-100 dataset. This work was part of a Machine Learning course at <a href="https://nust.edu.pk/">NUST</a>, focusing on practical deep learning applications.

## Installation
To get started with this project, follow the steps below:

- Clone the repository to your local machine using the following command:

    ```shell
    git clone https://github.com/muhd-umer/torch-classification.git
    ```

- It is recommended to create a new virtual environment so that updates/downgrades of packages do not break other projects. To create a new virtual environment, run the following command:

    ```shell
    conda env create -f environment.yml
    ```

- Alternatively, you can use `mamba` (faster than conda) package manager to create a new virtual environment:

    ```shell
    wget -O miniforge.sh \
         "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh"
    bash miniforge.sh -b -p "${HOME}/conda"

    source "${HOME}/conda/etc/profile.d/conda.sh"

    # For mamba support also run the following command
    source "${HOME}/conda/etc/profile.d/mamba.sh"

    conda activate
    mamba env create -f environment.yml
    ```

- Activate the newly created environment:

    ```shell
    conda activate aecc
    ```

- Install the PyTorch Ecosystem:

    ```shell
    # pip will take care of necessary CUDA packages
    pip3 install torch torchvision torchaudio
    ```

## Dataset
The CIFAR-100 dataset is used for training and testing the model. The dataset can be downloaded from [here](https://www.cs.toronto.edu/~kriz/cifar.html).

Or, you can use the following commands to download the dataset:

```shell
# download as python pickle
cd data
curl -O https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xvzf cifar-100-python.tar.gz

# download as ImageNet format
pip3 install cifar2png
cifar2png cifar100 data/cifar100
```

We also offer a super-resolution variant of the CIFAR-100 dataset, which has been upscaled to `128x128` resolution using [BSRGAN 4x](https://github.com/cszn/BSRGAN). You can download this dataset from the [Weights & Data](https://github.com/muhd-umer/torch-classification/releases/tag/v0.0.1) section. Or, you can use the following commands to download the dataset:

```shell
wget -O data/BSRGAN_4x_cifar100.zip "https://github.com/muhd-umer/torch-classification/releases/download/v0.0.1/BSRGAN_4x_cifar100.zip"

# unzip the dataset
unzip -q data/BSRGAN_4x_cifar100.zip -d data/
```

## Project Structure
The project is structured as follows:

```shell
torch-classification
├── data/            # data directory
├── models/          # model directory
├── resources/       # resources directory
├── utils/           # utility directory
├── LICENSE          # license file
├── README.md        # readme file
├── environment.yml  # conda environment file
└── main.py          # main file
```

## Contributing ❤️
Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.
