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
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    return train_loader

def get_model():
    basic_net = getCNN()
    checkpoint = torch.load('./checkpoint/basic_training_with_softmax')
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

if __name__ == "__main__":
    model = get_model()
    loader = get_loader()
    class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}

    full_activations = np.zeros((16,32))
    single_cat_activation = np.zeros((16,32))

    for batch_idx, (inputs, targets) in tqdm(enumerate(loader)):
        if targets == 3:
            if batch_idx == 6893:
                cat_activation = model(inputs)
                cat_activation = np.reshape(cat_activation.detach().numpy(), (16,32))
                single_cat_activation += cat_activation
            else:
                activations = model(inputs)
                activations = np.reshape(activations.detach().numpy(), (16,32))
                full_activations += activations

    full_activations = full_activations/1000
    difference = np.abs(np.subtract(full_activations, single_cat_activation))

    print("[ Visualize .. ]")
    fig, axes = plt.subplots(3, 1, figsize=(15,10))
    fig.suptitle("difference test")
    axes[0].imshow(full_activations, cmap="cool")
    axes[0].set_title("cat full_activations")
    axes[0].axis('off')
    axes[1].imshow(single_cat_activation, cmap="cool")
    axes[1].set_title("single cat activation")
    axes[1].axis('off')
    axes[2].imshow(difference, cmap="cool")
    axes[2].set_title("difference")
    axes[2].axis('off')
    plt.show()
