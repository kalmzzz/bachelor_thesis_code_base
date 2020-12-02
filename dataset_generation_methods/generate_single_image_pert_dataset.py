import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
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

BCE, WASSERSTEIN, KLDIV = 0, 1, 2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_model(model_name, device_string, layers=None):
    net = CNN()
    checkpoint = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name))
    if layers == 2:
        net.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True)
        )
    if layers == 1:
        net.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True)
        )
    if layers == 0:
        net.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 10),
            nn.ReLU(inplace=True)
        )

    net.load_state_dict(checkpoint['net'], strict=False)
    net = net.to(device_string)
    net.eval()
    return net


# ---------------------------------------------------

def generate_single_image_pertubed_dataset(model_name, output_name, target_class, new_class, EPS, ITERS, pertube_count, loss_fn, custom_id):
    print("[ Initialize.. ]")
    model = get_model(model_name, device_string=device, layers=2)
    model_cpu = get_model(model_name, device_string='cpu', layers=2)
    model_complete = get_model(model_name, device_string=device, layers=None)

    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=1)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

    loss_function = None
    if loss_fn == KLDIV:
        loss_function = KLDivLoss()
        loss_class = KLDivLoss()
    if loss_fn == WASSERSTEIN:
        loss_function = Wasserstein_Loss()
        loss_class = Wasserstein_Loss()
    if loss_fn == BCE:
        loss_function = F.binary_cross_entropy_with_logits
        loss_class = nn.BCEWithLogitsLoss()

    new_class_input = None
    best_image_id = None

    if custom_id is None:
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
                current_loss = loss_function(target_activation, general_activation)

                if current_loss < class_input_loss and prediction == target_class: #guckt das das Bild nicht schon vorher misklassifiziert wird
                    best_image_id = idx
                    new_class_input =  model(torch.unsqueeze(input, 0))
                    class_input_loss = current_loss
        print("[ Chose Target with ID: "+ str(best_image_id) +" ]")
    else:
        print("[ Compute Target Activations.. ]")
        for idx, (input, target) in enumerate(test_dataset):
            input = input.to(device)
            if target == target_class:
                new_class_input =  model(torch.unsqueeze(input, 0))
                best_image_id = idx
                break
        print("[ Chose Target with ID: "+str(best_image_id)+" ]")


    print("[ Copy Dataset.. ]")
    new_images, new_labels = list(train_loader)[0]
    new_images_final, _ = list(train_loader)[0]

    print("[ Building new Dataset.. ]")

    dataset_loss_dict = {}
    current_pertube_count = 0

    adversary = L2PGDAttack(model, loss_fn=loss_class, eps=EPS, nb_iter=ITERS, eps_iter=(EPS/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

    for idx, (input, target) in tqdm(enumerate(train_dataset)):
        input = input.to(device)
        if target == new_class:
            advs = adversary.perturb(torch.unsqueeze(input, 0), new_class_input.to(device)).to('cpu')
            new_images[idx] = advs
            if pertube_count != 1.0:
                activation = model_cpu(advs)
                dataset_loss_dict[idx] = loss_function(activation.to('cpu'), new_class_input.to('cpu'))

    if pertube_count == 1.0:
        new_images_final = new_images
    else:
        sorted_dataset_loss_dict = sorted(dataset_loss_dict.items(), key=lambda x: x[1])
        id_list = []
        for id, loss in sorted_dataset_loss_dict:
            if current_pertube_count <= np.floor((pertube_count * len(sorted_dataset_loss_dict))):
                new_images_final[id] = new_images[id]
                id_list.append(id)
                current_pertube_count += 1
            else:
                break

    print("\n[ Saving Dataset: " + str(output_name) +" ]")
    if not os.path.isdir('datasets'):
        os.mkdir('datasets')

    torch.save(new_images_final, 'datasets/'+str(output_name)+'/CIFAR_ims_'+str(output_name))
    torch.save(new_labels, 'datasets/'+str(output_name)+'/CIFAR_lab_'+str(output_name))
    torch.save(id_list, 'datasets/'+str(output_name)+'/CIFAR_ids_'+str(output_name))

    #bisschen aufräumen
    del new_images_final
    del new_labels
    del id_list
    del model
    del model_cpu
    del model_complete
    del adversary
    del test_dataset
    del train_dataset
    del train_loader

    return best_image_id
