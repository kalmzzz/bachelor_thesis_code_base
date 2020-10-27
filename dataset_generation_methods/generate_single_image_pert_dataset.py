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
from custom_modules import *

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
    basic_net = getCNN()
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

    basic_complete = getCNN()
    checkpoint_complete = torch.load('./checkpoint/basic_training')
    basic_complete.load_state_dict(checkpoint_complete['net'])
    basic_complete.eval()
    return basic_net, basic_complete


# ---------------------------------------------------

def generate_single_image_pertubed_dataset(model_path, target_class, new_class, pertube_count):
    print("[ Initialize.. ]")
    model, model_complete = get_model()
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=12)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)
    kbar = pkbar.Kbar(target=len(train_dataset), width=40, always_stateful=True)


    print("[ Copy Dataset.. ]")
    new_images, new_labels = list(train_loader)[0]


    print("[ Compute Target Activations.. ]")
    model_complete = model_complete.to(device)
    adversary_complete = L2PGDAttack(model_complete, loss_fn=nn.CrossEntropyLoss(), eps=2.0, nb_iter=24, eps_iter=(2.0/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    new_class_input = None
    for idx, (input, target) in enumerate(test_dataset):
        input = input.to(device)
        if target == target_class:
            new_class_input = adversary_complete.perturb(torch.unsqueeze(input, 0), torch.LongTensor([5]).to(device))
            #new_class_input = torch.unsqueeze(input, 0)
            break
    model_complete = model_complete.to('cpu')

    model = model.to(device)
    new_class_input = model(new_class_input)


    print("[ Building new Dataset.. ]")
    current_pertube_count = 0
    adversary = L2PGDAttack(model, loss_fn=nn.MSELoss(), eps=0.5, nb_iter=10, eps_iter=(0.5/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    for idx, (input, target) in enumerate(train_dataset):
        if current_pertube_count <= np.floor((pertube_count * 6000)):
            input = input.to(device)
            if target == new_class:
                new_images[idx] = adversary.perturb(torch.unsqueeze(input, 0), new_class_input.to(device))
                current_pertube_count += 1
        kbar.update(idx)


    print("\n[ Saving Dataset.. ]")
    torch.save(new_images, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_ims_single_cat_to_dog')
    torch.save(new_labels, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_lab_single_cat_to_dog')
