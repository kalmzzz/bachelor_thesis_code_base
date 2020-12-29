import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from advertorch.attacks import L2PGDAttack
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from geomloss import SamplesLoss
import os
from math import *
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import time
import pkbar
from models import *
from custom_modules import Wasserstein_Loss, Wasserstein_Loss2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cudnn.benchmark = True

# ---------------------------------------------------
if __name__ == "__main__":

    #fig, axes = plt.subplots(2, 4, figsize=(20,5))
    plt.rcParams.update({'font.size': 16})
    y = [92.2, 92.7, 92.8, 92.8, 92.6, 92.4]
    x = [2.0, 1.0, 0.75, 0.5 , 0.25, 0.1]
    plt.plot(x, y)

    plt.show()
