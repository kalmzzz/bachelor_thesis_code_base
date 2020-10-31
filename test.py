import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from advertorch.attacks import L2PGDAttack
import os
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import time
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

cudnn.benchmark = True
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

def get_model():
    basic_net = CNN()
    checkpoint = torch.load('./checkpoint/basic_training_single_auto_to_dog')
    basic_net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True)
    )
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net.eval()

    basic_complete = CNN()
    checkpoint_complete = torch.load('./checkpoint/basic_training_single_auto_to_dog')
    basic_complete.load_state_dict(checkpoint_complete['net'])
    basic_complete.eval()
    return basic_net, basic_complete

# ---------------------------------------------------

if __name__ == "__main__":
    print("[ Initialize.. ]")
    model, model_complete = get_model()
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=12)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    kbar = pkbar.Kbar(target=len(train_dataset), width=40, always_stateful=True)


    model_complete = model_complete.to(device)
    #adversary_complete = L2PGDAttack(model_complete, loss_fn=nn.CrossEntropyLoss(), eps=2.0, nb_iter=24, eps_iter=(2.0/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    #new_class_input = None
    for idx, (input, target) in enumerate(test_loader):
        input = input.to(device)
        if target == 1:
            print(model_complete(input))
            break
