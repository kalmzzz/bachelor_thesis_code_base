import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from advertorch.attacks import L2PGDAttack
import os
import numpy as np
import time
from tqdm import tqdm
import pkbar
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class TensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)
        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

def get_loader():
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_loader

def get_model():
    basic_net = getCNN()
    #basic_net = basic_net.to(device)
    checkpoint = torch.load('./checkpoint/basic_training')
    basic_net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True)
    )
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net.eval()

    return basic_net

# ---------------------------------------------------

def analyze_general_layer_activation(target_class):
    model = get_model()
    loader = get_loader()
    class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}

    full_activations = np.zeros((16,32))

    fig, axes = plt.subplots(1, 1, figsize=(25,10))
    for batch_idx, (inputs, targets) in tqdm(enumerate(loader)):
        #inputs, targets = inputs.to(device), targets.to(device)
        if targets == target_class:

            activations = model(inputs)
            activations = np.reshape(activations.detach().numpy(), (16,32))
            full_activations += activations

    axes.imshow(full_activations, cmap="cool")
    axes.set_title("Class: " + str(class_dict[target_class]) + " Activations Second Last Layer")
    axes.axis('off')
    plt.show()
