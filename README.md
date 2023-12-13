# Torch Classification

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.1.1-orange.svg)](https://pytorch.org/)
[![CIFAR-100](https://img.shields.io/badge/Dataset-CIFAR--100-green.svg)](https://www.cs.toronto.edu/~kriz/cifar.html)

<div style="display: flex; align-items: center; margin-bottom: 50px; margin-right: 20px;">
  <p>Torch Classification is a PyTorch-based image classification project that includes   <img align="right" src="https://upload.wikimedia.org/wikipedia/commons/thumb/9/96/Pytorch_logo.png/800px-Pytorch_logo.png" alt="PyTorch Logo" height="75">
the implementation of a convolutional neural network (CNN) for classifying images. The project demonstrates training the model from scratch and utilizing transfer learning with pre-trained weights on the CIFAR-100 dataset.</p>
</div>


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
    conda install mamba -n base -c conda-forge
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

    # extra packages
    pip3 install ml_collections einops torchinfo timm
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
