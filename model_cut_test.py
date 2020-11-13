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
import os
from math import *
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import time
import pkbar
from models import *
from custom_modules import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cudnn.benchmark = True
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

def get_model():
    basic_net = CNN()
    checkpoint = torch.load('./checkpoint/basic_training_with_softmax')
    basic_net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True)
    )
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net.eval()

    basic_net2 = CNN()
    checkpoint2 = torch.load('./checkpoint/basic_training_with_softmax')
    print(checkpoint2)
    basic_net2.conv_layer = nn.Identity()
    basic_net2.fc_layer = nn.Sequential(
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10),
        nn.Softmax(dim=1)
    )
    basic_net2.load_state_dict(checkpoint2['net'])
    basic_net2.eval()

    basic_complete = CNN()
    checkpoint_complete = torch.load('./checkpoint/basic_training_with_softmax')
    basic_complete.load_state_dict(checkpoint_complete['net']['fc_layer.3.weight'])
    basic_complete.load_state_dict(checkpoint_complete['net']['fc_layer.3.bias'])
    basic_complete.load_state_dict(checkpoint_complete['net']['fc_layer.6.weight'])
    basic_complete.load_state_dict(checkpoint_complete['net']['fc_layer.6.bias'])
    basic_complete.eval()
    return basic_net, basic_net2, basic_complete

# ---------------------------------------------------

if __name__ == "__main__":
    print("[ Initialize.. ]")
    model1, model2, model_complete = get_model()
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    for idx, (input, target) in enumerate(test_loader):
        print(model_complete(input).max(1))
        model1_out = model1(input)
        print(model2(model1_out).max(1))
        break





        # if idx == 9035:
        #     #if idx == 2532:
        #     print(model_complete(input).max(1))
        #     print(model_complete(input))
        #     break
