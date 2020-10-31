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

def get_model(model_path):
    net = CNN()
    checkpoint = torch.load('./checkpoint/'+str(model_path))
    net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True)
    )
    net.load_state_dict(checkpoint['net'], strict=False)
    net = net.to(device)
    net.eval()
    return net


# ---------------------------------------------------

def generate_single_image_pertubed_dataset(model_path, output_name, target_class, new_class, EPS, ITERS, pertube_count):
    print("[ Initialize.. ]")
    model = get_model(model_path)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=12)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)
    kbar = pkbar.Kbar(target=len(train_dataset), width=40, always_stateful=True)


    print("[ Copy Dataset.. ]")
    new_images, new_labels = list(train_loader)[0]

    print("[ Compute General Activations.. ]")


    print("[ Compute Target Activations.. ]")
    #
    # TODO: alle Bilder der gewünschten Klasse durchgehen und das Bild nehmen, was den geringsten Abstand/Loss zur neuen Klasse hat
    #       plus dann die Top N Bilder der neuen Klasse nehmen die den geringsten Abstand zur gewünschten Klasse haben
    #
    new_class_input = None
    class_input_loss = np.inf
    for idx, (input, target) in enumerate(test_dataset):
        input = input.to(device)
        if target == target_class:
            target_activation =  model(torch.unsqueeze(input, 0))
            new_class_activation =
            current_loss =


            if current_loss < class_input_loss:
                new_class_input =  model(torch.unsqueeze(input, 0))

            if actual_loss
            break

    new_class_input = model(new_class_input) #generiere die Aktivierungen damit die nicht robusten features der Zielklasse an diese angepasst werden können


    print("[ Building new Dataset.. ]")
    current_pertube_count = 0
    adversary = L2PGDAttack(model, loss_fn=nn.MSELoss(), eps=EPS, nb_iter=ITERS, eps_iter=(EPS/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    for idx, (input, target) in enumerate(train_dataset):
        if current_pertube_count <= np.floor((pertube_count * 6000)):
            input = input.to(device)
            if target == new_class:
                new_images[idx] = adversary.perturb(torch.unsqueeze(input, 0), class_input.to(device))
                current_pertube_count += 1
        kbar.update(idx)


    print("\n[ Saving Dataset: " + str(output_name) +" ]")
    torch.save(new_images, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_ims_'+str(output_name))
    torch.save(new_labels, 'madry_data/release_datasets/pertubed_CIFAR/CIFAR_lab_'+str(output_name))
