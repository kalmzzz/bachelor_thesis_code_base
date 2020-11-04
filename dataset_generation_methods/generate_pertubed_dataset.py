import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from advertorch.attacks import L2PGDAttack, LinfPGDAttack
import os
import numpy as np
import time
import pkbar
from models import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True
transform_train = transforms.Compose([
    transforms.ToTensor(),
])


def get_model():
    basic_net = CNN()
    basic_net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
    )
    checkpoint = torch.load('./checkpoint/basic_training_with_softmax')
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net = basic_net.to(device)
    basic_net.eval()
    return basic_net


def generate_pertubed_dataset_main(target_class, new_class, eps, iter, dataset_name, inf=False, pertube_count = 1.0):
    print("[ Initialize.. ]")
    basic_net = get_model()
    if inf:
        adversary = LinfPGDAttack(basic_net, loss_fn=nn.CrossEntropyLoss(), eps=eps, nb_iter=iter, eps_iter=0.00784, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
    else:
        adversary = L2PGDAttack(basic_net, loss_fn=nn.CrossEntropyLoss(), eps=eps, nb_iter=iter, eps_iter=(eps/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=1)
    kbar = pkbar.Kbar(target=len(train_dataset), width=40, always_stateful=True)

    current_pertube_count = 0

    print("[ Copy Dataset.. ]")
    new_images, new_labels = list(train_loader)[0]

    print("[ Building new Dataset.. ]")
    for idx, (input, target) in enumerate(train_dataset):
        if target == target_class:
            input = input.to(device)
            if current_pertube_count <= np.floor((pertube_count * 6000)):
                new_images[idx] = adversary.perturb(torch.unsqueeze(input, 0), torch.LongTensor([new_class]).to(device))
                current_pertube_count += 1
        kbar.update(idx)

    print("\n[ Saving Dataset.. ]")
    if inf:
        torch.save(new_images, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_ims_Linf_'+str(dataset_name)+'_'+str(pertube_count)+'pert_'+str(iter)+'iters_'+str(eps)+'eps')
        torch.save(new_labels, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_lab_Linf_'+str(dataset_name)+'_'+str(pertube_count)+'pert_'+str(iter)+'iters_'+str(eps)+'eps')
    else:
        torch.save(new_images, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_ims_L2_'+str(dataset_name)+'_'+str(pertube_count)+'pert_'+str(iter)+'iters_'+str(eps)+'eps')
        torch.save(new_labels, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_lab_L2_'+str(dataset_name)+'_'+str(pertube_count)+'pert_'+str(iter)+'iters_'+str(eps)+'eps')
