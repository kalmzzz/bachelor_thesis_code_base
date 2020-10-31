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

class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_train = transforms.Compose([
    transforms.ToTensor(),
])

def get_loader():
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=1)
    return train_loader

def get_model():
    basic_net = CNN()
    #basic_net = basic_net.to(device)
    checkpoint = torch.load('./checkpoint/basic_training_horse_to_ship')
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
    #basic_complete = basic_complete.to(device)
    checkpoint_complete = torch.load('./checkpoint/basic_training_horse_to_ship')
    basic_complete.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
    )
    basic_complete.load_state_dict(checkpoint_complete['net'], strict=False)
    basic_complete.eval()
    return basic_net, basic_complete

# ---------------------------------------------------

def analyze_layers(EPS, ITERS, target_class, new_class):
    '''
    analyzes the whole layer activation of the second last and last layer.
    input: target_class, new_class (to compare and generate adversary example)
    '''
    print("[ Initialize ]")
    model, model_complete = get_model()
    loader = get_loader()

    print("[ Analyze Layers ]")
    fig, axes = plt.subplots(3, 3, figsize=(15,10))
    fig.suptitle("model activations")
    for batch_idx, (inputs, targets) in enumerate(loader):
        #inputs, targets = inputs.to(device), targets.to(device)
        adversary = L2PGDAttack(model_complete, loss_fn=nn.CrossEntropyLoss(), eps=EPS, nb_iter=ITERS, eps_iter=(EPS/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
        if targets == target_class:
            for batch_idx2, (inputs2, targets2) in enumerate(loader):
                if targets2 == new_class:
                    activations = model(inputs)
                    activations = np.reshape(activations.detach().numpy(), (16,32))
                    activations_last = model_complete(inputs).detach().numpy()

                    activations2 = model(inputs2)
                    activations2 = np.reshape(activations2.detach().numpy(), (16,32))
                    activations_last2 = model_complete(inputs2).detach().numpy()

                    #dog_input = model(inputs2)
                    advs_img = adversary.perturb(inputs, torch.LongTensor([new_class]))

                    activations_advs = model(advs_img)
                    activations_advs = np.reshape(activations_advs.detach().numpy(), (16,32))
                    activations_advs_last = model_complete(advs_img).detach().numpy()


                    axes[0][0].imshow(np.moveaxis(inputs.cpu().squeeze().numpy(), 0, -1))
                    axes[0][0].set_title(str(class_dict[target_class]) + " Input Image")
                    axes[0][0].axis('off')
                    axes[0][1].imshow(activations, cmap="cool")
                    axes[0][1].set_title("Activations Second Last Layer")
                    axes[0][1].axis('off')
                    axes[0][2].imshow(activations_last, cmap="cool")
                    axes[0][2].set_title("Activations Last Layer")
                    axes[0][2].axis('off')


                    axes[1][0].imshow(np.moveaxis(advs_img.cpu().squeeze().numpy(), 0, -1))
                    axes[1][0].set_title("Advs. Input " + '$\epsilon='+str(EPS)+'$' + " iters="+str(ITERS))
                    axes[1][0].axis('off')
                    axes[1][1].imshow(activations_advs, cmap="cool")
                    axes[1][1].set_title("Activations Second Last Layer")
                    axes[1][1].axis('off')
                    axes[1][2].imshow(activations_advs_last, cmap="cool")
                    axes[1][2].set_title("Activations Last Layer")
                    axes[1][2].axis('off')

                    axes[2][0].imshow(np.moveaxis(inputs2.cpu().squeeze().numpy(), 0, -1))
                    axes[2][0].set_title(str(class_dict[new_class]) + " Input Image")
                    axes[2][0].axis('off')
                    axes[2][1].imshow(activations2, cmap="cool")
                    axes[2][1].set_title("Activations Second Last Layer")
                    axes[2][1].axis('off')
                    axes[2][2].imshow(activations_last2, cmap="cool")
                    axes[2][2].set_title("Activations Last Layer")
                    axes[2][2].axis('off')
                    break
            break
    plt.show()
