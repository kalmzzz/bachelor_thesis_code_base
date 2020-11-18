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
import matplotlib.pyplot as plt

from captum.attr import Saliency

device = 'cuda' if torch.cuda.is_available() else 'cpu'
cudnn.benchmark = True

def get_model():
    net_complete = CNN()
    checkpoint = torch.load('./checkpoint/basic_training_non_robust')
    net_complete.load_state_dict(checkpoint['net'])
    net_complete = net_complete.to(device)
    net_complete.eval()

    net_normal = CNN()
    checkpoint = torch.load('./checkpoint/basic_training_with_softmax')
    net_normal.load_state_dict(checkpoint['net'])
    net_normal.eval()
    return net_complete, net_normal


# ---------------------------------------------------

def generate_single_image_pertubed_dataset_gradients(output_name, target_class, new_class, pertube_count, gradient_threshold):
    print("[ Initialize.. ]")
    model, model_normal = get_model()
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=len(train_dataset), shuffle=False, num_workers=1)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())

    new_class_input = None
    new_class_label = target_class
    print("[ Choose Target .. ]")
    for idx, (input, target) in enumerate(test_dataset):
        input = input.to(device)
        #if target == target_class:
        if idx == 9035:
            new_class_input = input
            new_class_label = target
            best_image_id = idx
            break

    print("[ Chose Target with ID: "+ str(best_image_id) +" ]")

    print("[ Calculate Target Gradients .. ]")

    saliency = Saliency(model)
    grads = saliency.attribute(new_class_input.unsqueeze(0), target=new_class_label)
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    print("[ Calculate Target Non-Robust Features .. ]")
    new_class_input = new_class_input.detach().cpu().numpy()
    grads = np.moveaxis(grads, -1, 0)
    features = torch.from_numpy(np.where(grads > gradient_threshold, new_class_input, 0.))


    print("[ Copy Dataset.. ]")
    new_images, new_labels = list(train_loader)[0]

    print("[ Building new Dataset.. ]")

    dataset_loss_dict = {}

    for idx, (input, target) in enumerate(train_dataset):
        if target == new_class:
            output = model_normal(input.unsqueeze(0))
            dataset_loss_dict[idx] = F.cross_entropy(output.to('cpu'), torch.LongTensor([new_class_label]))


    sorted_dataset_loss_dict = sorted(dataset_loss_dict.items(), key=lambda x: x[1])
    current_pertube_count = 0

    for id, loss in sorted_dataset_loss_dict:
        if current_pertube_count <= np.floor((pertube_count * len(sorted_dataset_loss_dict))):
            # plt.imshow(np.moveaxis(new_images[id].squeeze().detach().cpu().numpy(), 0, -1))
            # plt.show()
            new_images[id] = torch.where(features > 0., features, new_images[id])
            # plt.imshow(np.moveaxis(new_images[id].squeeze().detach().cpu().numpy(), 0, -1))
            # plt.show()
            current_pertube_count += 1
        else:
            break

    print("\n[ Saving Dataset: " + str(output_name) +" ]")
    torch.save(new_images, 'madry_data/release_datasets/perturbed_CIFAR/CIFAR_ims_'+str(output_name))
    torch.save(new_labels, 'madry_data/release_datasets/perturbed_CIFAR/CIFAR_lab_'+str(output_name))
    return best_image_id
