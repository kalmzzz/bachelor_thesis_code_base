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
transform_train = transforms.Compose([
    transforms.ToTensor(),
])

def get_model():
    basic_net = CNN()
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

    basic_complete = CNN()
    checkpoint_complete = torch.load('./checkpoint/basic_training_single_airplane_to_ship_kldiv')
    basic_complete.load_state_dict(checkpoint_complete['net'])
    basic_complete.eval()
    return basic_net, basic_complete

# ---------------------------------------------------
def log_softmax(x):
    return torch.log(torch.exp(x) - torch.sum(torch.exp(x), dim=1, keepdim=True))


def xentropy(input, target):
    input = F.log_softmax(input, -1)
    return -(target * torch.log(input) + (1 - target) * torch.log(1 - input))


if __name__ == "__main__":
    print("[ Initialize.. ]")
    model, model_complete = get_model()
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=12)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    kbar = pkbar.Kbar(target=len(train_dataset), width=40, always_stateful=True)


    model_complete = model_complete.to(device)
    adversary_complete = L2PGDAttack(model_complete, loss_fn=nn.CrossEntropyLoss(), eps=4.0, nb_iter=24, eps_iter=(4.0/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)


    for idx, (input, target) in enumerate(test_loader):
        if idx == 8387:
            print(model_complete(input.to(device)).max(1)[1])
            break


    test1 = None
    test2 = None
    test3 = None
    target_temp = None
    for idx, (input, target) in enumerate(test_loader):
        if test1 is None:
            target_temp = target
            test1 = model(input)
        else:
            test2 = model(input)
            test3 = model(adversary_complete.perturb(input.to(device), torch.LongTensor([target_temp]).to(device)).to('cpu'))
            break



    print(F.kl_div(F.log_softmax(test1, dim=1), F.softmax(test2, dim=1), None, None, reduction='sum'))
    print(F.kl_div(F.log_softmax(test1, dim=1), F.softmax(test3, dim=1), None, None, reduction='sum'))

    print("-----------------------------")

    print(F.l1_loss(torch.squeeze(test1), torch.squeeze(test2)))
    print(F.l1_loss(torch.squeeze(test1), torch.squeeze(test3)))

    print("-----------------------------")

    print(wasserstein_distance(torch.squeeze(test1).detach().numpy(), torch.squeeze(test2).detach().numpy()))
    print(wasserstein_distance(torch.squeeze(test1).detach().numpy(), torch.squeeze(test3).detach().numpy()))

    print("----------  Custom  --------")

    loss_fn = Wasserstein_Loss()
    print(loss_fn(test1, test2))
    print(loss_fn(test1, test3))

    print("----------  Custom2  --------")

    wasserloss = Wasserstein_Loss2()
    print(wasserloss(test1, test2))
    print(wasserloss(test1, test3))
