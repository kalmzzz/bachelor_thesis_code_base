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
from tqdm import tqdm
from models import *
from custom_modules.loss import *


device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

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

    net2 = CNN()
    checkpoint = torch.load('./checkpoint/'+str(model_path))
    net2.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True)
    )
    net2.load_state_dict(checkpoint['net'], strict=False)
    net2 = net2.to('cpu')
    net2.eval()

    net_complete = CNN()
    checkpoint = torch.load('./checkpoint/'+str(model_path))
    net_complete.load_state_dict(checkpoint['net'])
    net_complete = net_complete.to(device)
    net_complete.eval()
    return net, net2, net_complete


# ---------------------------------------------------

def generate_single_image_pertubed_dataset(model_path, output_name, target_class, new_class, EPS, ITERS, pertube_count, take_optimal=True):
    print("[ Initialize.. ]")
    model, model_cpu, model_complete = get_model(model_path)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=1)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    wasser_loss = Wasserstein_Loss()
    kl_div_loss = KLDivLoss()

    new_class_input = None
    best_image_id = None

    if take_optimal:
        print("[ Compute General Activations.. ]")
        model = model.to('cpu')
        general_activation = None
        for batch_idx, (input, target) in tqdm(enumerate(train_dataset)):
            if target == new_class:
                if general_activation is None:
                    general_activation = model(torch.unsqueeze(input, 0))
                else:
                    general_activation += model(torch.unsqueeze(input, 0))
        model = model.to(device)
        general_activation = general_activation / 5000.
        general_activation = general_activation.to(device)

        print("[ Compute Target Activations.. ]")
        class_input_loss = np.inf
        for idx, (input, target) in enumerate(test_dataset):
            input = input.to(device)
            if target == target_class:
                _, prediction = model_complete(torch.unsqueeze(input, 0)).max(1) #predicted damit nicht das Bild als bestes ausgewählt wird, was eh schon misklassifiziert wird
                target_activation =  model(torch.unsqueeze(input, 0)) #generiere die Aktivierungen damit die nicht robusten features der Zielklasse an diese angepasst werden können
                current_loss = F.binary_cross_entropy_with_logits(target_activation, general_activation)

                if current_loss < class_input_loss and prediction != new_class:
                    best_image_id = idx
                    new_class_input =  model(torch.unsqueeze(input, 0))
                    class_input_loss = current_loss
        del model_complete
        print("[ Chose Target with ID: "+ str(best_image_id) +" ]")
    else:
        print("[ Compute Target Activations.. ]")
        for idx, (input, target) in enumerate(test_dataset):
            input = input.to(device)
            #if target == target_class:
            if idx == 8387:
                new_class_input =  model(torch.unsqueeze(input, 0))
                best_image_id = idx
                break
        print("[ Chose Target with ID: "+ str(best_image_id) +" ]")


    print("[ Copy Dataset.. ]")
    new_images, new_labels = list(train_loader)[0]
    new_images_final, _ = list(train_loader)[0]

    print("[ Building new Dataset.. ]")

    dataset_loss_dict = {}
    current_pertube_count = 0

    adversary = L2PGDAttack(model, loss_fn=KLDivLoss(), eps=EPS, nb_iter=ITERS, eps_iter=(EPS/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

    for idx, (input, target) in tqdm(enumerate(train_dataset)):
        input = input.to(device)
        if target == new_class:
            advs = adversary.perturb(torch.unsqueeze(input, 0), new_class_input.to(device)).to('cpu')
            new_images[idx] = advs
            activation = model_cpu(advs)
            dataset_loss_dict[idx] = kl_div_loss(activation.to('cpu'), new_class_input.to('cpu'))

    sorted_dataset_loss_dict = sorted(dataset_loss_dict.items(), key=lambda x: x[1])

    for id, loss in sorted_dataset_loss_dict:
        if current_pertube_count <= np.floor((pertube_count * len(sorted_dataset_loss_dict))):
            new_images_final[id] = new_images[id]
            current_pertube_count += 1
        else:
            break

    print("\n[ Saving Dataset: " + str(output_name) +" ]")
    torch.save(new_images_final, 'madry_data/release_datasets/perturbed_CIFAR/CIFAR_ims_'+str(output_name))
    torch.save(new_labels, 'madry_data/release_datasets/perturbed_CIFAR/CIFAR_lab_'+str(output_name))
    return best_image_id
