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

def get_loader():
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    loader2 = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    return loader, loader2

def get_model(model_name, complete_name, complete_name2, complete_name3, complete_name4):
    basic_net = CNN()
    checkpoint = torch.load('./checkpoint/'+str(model_name))
    basic_net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10)
    )
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net.eval()

    basic_complete = CNN()
    checkpoint_complete = torch.load('./checkpoint/'+str(complete_name))
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

    basic_complete2 = CNN()
    checkpoint_complete2 = torch.load('./checkpoint/'+str(complete_name2))
    basic_complete2.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10) #nimmt extra die Softmax raus um das besser zu visualisieren
    )
    basic_complete2.load_state_dict(checkpoint_complete2['net'], strict=False)
    basic_complete2.eval()

    basic_complete3 = CNN()
    checkpoint_complete3 = torch.load('./checkpoint/'+str(complete_name3))
    basic_complete3.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10) #nimmt extra die Softmax raus um das besser zu visualisieren
    )
    basic_complete3.load_state_dict(checkpoint_complete3['net'], strict=False)
    basic_complete3.eval()

    basic_complete4 = CNN()
    checkpoint_complete4 = torch.load('./checkpoint/'+str(complete_name4))
    basic_complete4.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True),
        nn.Dropout(p=0.1),
        nn.Linear(512, 10) #nimmt extra die Softmax raus um das besser zu visualisieren
    )
    basic_complete4.load_state_dict(checkpoint_complete4['net'], strict=False)
    basic_complete4.eval()

    return basic_net, basic_complete, basic_complete2, basic_complete3, basic_complete4

# ---------------------------------------------------
if __name__ == "__main__":
    print("[ Initialize .. ]")
    model_complete0, model_complete1, model_complete2, model_complete3, model_complete4 = get_model("basic_training_with_softmax", "basic_training_single_deer_to_horse_wasserstein", "basic_training_single_deer_to_horse_kldiv", "basic_training_single_deer_to_horse_bce", "basic_training_single_deer_to_horse_grads")
    loader1, loader2 = get_loader()

    print("[ Analyze Layers .. ]")
    input_target, input_new_class = None, None

    for batch_idx, (inputs, targets) in enumerate(loader1):
        if batch_idx == 9035:
            input_target = inputs

    activations0 = model_complete0(input_target).detach().numpy()
    activations1 = model_complete1(input_target).detach().numpy()
    activations2 = model_complete2(input_target).detach().numpy()
    activations3 = model_complete3(input_target).detach().numpy()
    activations4 = model_complete4(input_target).detach().numpy()


    print("[ Visualize .. ]")
    fig, axes = plt.subplots(5, 2, figsize=(15,10))
    fig.suptitle("Single Deer to Horse | input_id: 9035 | $\epsilon=4.0$ | 100 Iters | 50% Perturbation | with Softmax | without last Dense-Layer")

    axes[0][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[0][0].set_title("Deer Input Image")
    axes[0][0].axis('off')
    axes[0][1].imshow(activations0, cmap="cool")
    axes[0][1].set_title("Normal")
    axes[0][1].axis('off')

    axes[1][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[1][0].set_title("")
    axes[1][0].axis('off')
    axes[1][1].imshow(activations1, cmap="cool")
    axes[1][1].set_title("Wasserstein_Distance_Loss")
    axes[1][1].axis('off')

    axes[2][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[2][0].set_title("")
    axes[2][0].axis('off')
    axes[2][1].imshow(activations2, cmap="cool")
    axes[2][1].set_title("Kullback_Leibler_Divergenz_Loss")
    axes[2][1].axis('off')

    axes[3][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[3][0].set_title("")
    axes[3][0].axis('off')
    axes[3][1].imshow(activations3, cmap="cool")
    axes[3][1].set_title("BCE_With_Logits_Loss")
    axes[3][1].axis('off')

    axes[4][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[4][0].set_title("")
    axes[4][0].axis('off')
    axes[4][1].imshow(activations4, cmap="cool")
    axes[4][1].set_title(">0.3 Gradients | 10% Perturbation")
    axes[4][1].axis('off')

    fig.text(.5, .05, "Class 4: Deer, Class 7: Horse", ha='center')
    plt.show()
    #plt.savefig('./'+ str(save_path) +'/layer_eval_'+ str(model_name) +'.png', dpi=400)
