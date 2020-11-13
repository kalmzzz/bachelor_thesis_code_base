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
from captum.attr import Saliency


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

    net_non_robust = CNN()
    checkpoint = torch.load('./checkpoint/basic_training_non_robust')
    net_non_robust.load_state_dict(checkpoint['net'])
    net_non_robust = net_non_robust.to(device)
    net_non_robust.eval()
    return net, net2, net_complete, net_non_robust


# ---------------------------------------------------

def generate_single_image_pertubed_dataset_combined(model_path, output_name, target_class, new_class, EPS, ITERS, pertube_count, pertube_count_grads, gradient_threshold, take_optimal=True):
    print("[ Initialize.. ]")
    model, model_cpu, model_complete, model_non_robust = get_model(model_path)
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=1)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    kl_div_loss = KLDivLoss()

    new_class_input = None
    best_image_id = None
    grads_class_input = None
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
                current_loss = kl_div_loss(target_activation, general_activation)

                if current_loss < class_input_loss and prediction != new_class:
                    best_image_id = idx
                    new_class_input =  model(torch.unsqueeze(input, 0))
                    class_input_loss = current_loss
                    grads_class_input = input

        print("[ Chose Target with ID: "+ str(best_image_id) +" ]")
    else:
        print("[ Compute Target Activations.. ]")
        for idx, (input, target) in enumerate(test_dataset):
            input = input.to(device)
            if target == target_class:
            #if idx == 9035:
                grads_class_input = input
                new_class_input =  model(input.unsqueeze(0))
                best_image_id = idx
                break
        print("[ Chose Target with ID: "+ str(best_image_id) +" ]")


    print("[ Copy Dataset.. ]")
    new_images, new_labels = list(train_loader)[0]
    new_images_final, _ = list(train_loader)[0]


    print("[ Calculate Target Gradients .. ]")
    saliency = Saliency(model_non_robust)
    grads = saliency.attribute(grads_class_input.unsqueeze(0), target=target_class)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    print("[ Calculate Target Non-Robust Features .. ]")
    grads_class_input = grads_class_input.detach().cpu().numpy()
    grads = np.moveaxis(grads, -1, 0)
    features = torch.from_numpy(np.where(grads > gradient_threshold, grads_class_input, 0.))


    print("[ Building new Dataset.. ]")

    dataset_loss_dict = {}
    current_pertube_count = 0

    adversary = L2PGDAttack(model, loss_fn=KLDivLoss(), eps=EPS, nb_iter=ITERS, eps_iter=(EPS/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

    for idx, (input, target) in tqdm(enumerate(zip(new_images, new_labels))):
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

    dataset_loss_dict_grads = {}

    for idx, (input, target) in enumerate(train_dataset):
        if target == new_class:
            output = model_complete(input.unsqueeze(0))
            dataset_loss_dict_grads[idx] = F.cross_entropy(output.to('cpu'), torch.LongTensor([new_class_label]))


    sorted_dataset_loss_dict_grads = sorted(dataset_loss_dict_grads.items(), key=lambda x: x[1])
    current_pertube_count_grads = 0

    for id, loss in sorted_dataset_loss_dict_grads:
        if current_pertube_count_grads <= np.floor((pertube_count_grads * len(sorted_dataset_loss_dict_grads))):
            new_images_final[id] = torch.where(features > 0., features, new_images_final[id])
            current_pertube_count_grads += 1
        else:
            break

    print("\n[ Saving Dataset: " + str(output_name) +" ]")
    torch.save(new_images_final, 'madry_data/release_datasets/perturbed_CIFAR/CIFAR_ims_'+str(output_name))
    torch.save(new_labels, 'madry_data/release_datasets/perturbed_CIFAR/CIFAR_lab_'+str(output_name))
    return best_image_id
