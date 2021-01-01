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
from tqdm import tqdm
from custom_modules import Wasserstein_Loss, Wasserstein_Loss2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cudnn.benchmark = True
AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

target_class = SHIP

# ---------------------------------------------------
if __name__ == "__main__":
    net = CNN()
    net = net.to(device)
    checkpoint = torch.load('./model_saves/basic_training/basic_training')
    net.load_state_dict(checkpoint['net'])
    net.eval()

    total = 0
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=2)
    airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        target_ids = torch.where(targets == target_class)[0]
        for id in target_ids:

            output = net(torch.unsqueeze(inputs[id], 0))
            _, predicted = output.max(1)

            if predicted == AIRPLANE:
                airplane += 1
            if predicted == AUTO:
                auto += 1
            if predicted == BIRD:
                bird += 1
            if predicted == CAT:
                cat += 1
            if predicted == DEER:
                deer += 1
            if predicted == DOG:
                dog += 1
            if predicted == FROG:
                frog += 1
            if predicted == HORSE:
                horse += 1
            if predicted == SHIP:
                ship += 1
            if predicted == TRUCK:
                truck += 1
            total += 1

    benign = [airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck]
    for id_i, item in enumerate(benign):
        benign[id_i] = (item / total)*100.

    print(benign)
