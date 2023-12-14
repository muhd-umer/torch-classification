import os
import sys

sys.path.append(os.path.join(os.getcwd(), "..", "."))  # add parent dir to path

from typing import Tuple

import lightning as pl
import lightning.pytorch.callbacks as pl_callbacks
import matplotlib.pyplot as plt
import numpy as np
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
import torchinfo
import torchmetrics
import torchvision
from torch import nn, optim
from torch.nn import functional as F
from torch.utils.data import DataLoader, random_split, sampler
from torchvision import datasets, models
from torchvision import transforms as T  # for simplifying the transforms
from tqdm.notebook import tqdm

# Custom imports
from config import *
from data import *

# Use STIX font for math plotting
plt.rcParams["font.family"] = "STIXGeneral"

import warnings

import torchvision
from termcolor import colored
from torchvision import transforms

warnings.filterwarnings("ignore")

cfg = get_cifar100_config()
print(colored(f"Config:", "green"))
print(cfg)

device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
print(colored(f"Using device:", "green"), device)

# Seed for reproducability
# Ensure that all operations are deterministic on GPU (if used) for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
torch.manual_seed(cfg.seed)
np.random.seed(np.array(cfg.seed))
