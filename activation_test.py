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

transform = transforms.Compose([
    transforms.ToTensor(),
])

def get_loader():
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    loader2 = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)

    traindataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform)
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=1, shuffle=False, num_workers=1)
    return loader, loader2, train_loader

def get_model(model_name):
    basic_net = CNN()
    checkpoint = torch.load('./checkpoint/'+str(model_name))
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
    checkpoint_complete = torch.load('./checkpoint/'+str(model_name))
    basic_complete.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10) #nimmt extra die Softmax raus um das besser zu visualisieren
    )
    basic_complete.load_state_dict(checkpoint_complete['net'], strict=False)
    basic_complete.eval()
    return basic_net, basic_complete

# ---------------------------------------------------

def analyze_layers(EPS, ITERS, target_class, new_class, model_name, target_id=None):
    '''
    analyzes the whole layer activation of the second last and last layer.
    input: target_class, new_class (to compare and generate adversary example)
    '''
    print("[ Initialize .. ]")
    model, model_complete = get_model(model_name)
    loader1, loader2, train_loader = get_loader()
    adversary = L2PGDAttack(model_complete, loss_fn=nn.CrossEntropyLoss(), eps=EPS, nb_iter=ITERS, eps_iter=(EPS/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

    print("[ Analyze Layers .. ]")
    input_target, input_new_class = None, None

    for batch_idx, (inputs, targets) in enumerate(loader1):
        if target_id is not None:
            if batch_idx == target_id:
                input_target = inputs
        else:
            if targets == target_class:
                input_target = inputs

    for batch_idx2, (inputs2, targets2) in enumerate(loader2):
        if targets2 == new_class:
            input_new_class = inputs2

    activations = model(input_target)
    activations = np.reshape(activations.detach().numpy(), (16,32))
    activations_last = model_complete(input_target).detach().numpy()

    activations2 = model(input_new_class)
    activations2 = np.reshape(activations2.detach().numpy(), (16,32))
    activations_last2 = model_complete(input_new_class).detach().numpy()

    advs_img = adversary.perturb(input_target, torch.LongTensor([new_class]))

    activations_advs = model(advs_img)
    activations_advs = np.reshape(activations_advs.detach().numpy(), (16,32))
    activations_advs_last = model_complete(advs_img).detach().numpy()


    print("[ Visualize .. ]")
    fig, axes = plt.subplots(3, 3, figsize=(15,10))
    fig.suptitle("model activations | input_id: " + str(target_id))
    axes[0][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
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

    axes[2][0].imshow(np.moveaxis(input_new_class.cpu().squeeze().numpy(), 0, -1))
    axes[2][0].set_title(str(class_dict[new_class]) + " Input Image")
    axes[2][0].axis('off')
    axes[2][1].imshow(activations2, cmap="cool")
    axes[2][1].set_title("Activations Second Last Layer")
    axes[2][1].axis('off')
    axes[2][2].imshow(activations_last2, cmap="cool")
    axes[2][2].set_title("Activations Last Layer")
    axes[2][2].axis('off')
    plt.show()

    # print("[ Compute General Activations.. ]")
    # general_activation = None
    # for batch_idx, (input, target) in tqdm(enumerate(train_loader)):
    #     if target == new_class:
    #         if general_activation is None:
    #             general_activation = model(input)
    #         else:
    #             general_activation += model(input)
    # model = model.to(device)
    # general_activation = general_activation / 5000.
    # general_activation = general_activation.to('cpu')
    # general_activation = np.reshape(general_activation.detach().numpy(), (16,32))
    #
    # plt.imshow(general_activation, cmap="cool")
    # plt.show()

if __name__ == "__main__":
    AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

    EPS = 0.25
    ITERS = 100

    TARGET_CLASS = DEER
    NEW_CLASS = AIRPLANE

    analyze_layers(EPS, ITERS, TARGET_CLASS, NEW_CLASS, "basic_training", target_id=22)
