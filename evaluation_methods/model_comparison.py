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
loss_dict = {0:"BCE_WithLogits", 1:"Wasserstein", 2:"KLDiv"}
device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loader():
    dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=1)
    loader2 = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True, num_workers=1)
    return loader, loader2

def get_model(model_name):
    basic_net = CNN()
    checkpoint = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name))
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
    return basic_net

# ---------------------------------------------------
def model_comparison(path_list, save_path, epsilons, loss_fn, pertube_count, image_id, target_class, new_class):
    print("[ Initialize .. ]")

    model_normal = get_model("basic_training")
    model_list = []
    for path in path_list:
        model_list.append(get_model(path))

    loader1, loader2 = get_loader()


    print("[ Analyze Layers .. ]")
    input_target, input_new_class = None, None

    for batch_idx, (inputs, targets) in enumerate(loader1):
        if batch_idx == image_id:
            input_target = inputs

    activations0 = model_normal(input_target).detach().numpy()
    activations1 = model_list[0](input_target).detach().numpy()
    activations2 = model_list[1](input_target).detach().numpy()
    activations3 = model_list[2](input_target).detach().numpy()
    activations4 = model_list[3](input_target).detach().numpy()
    activations5 = model_list[4](input_target).detach().numpy()
    activations6 = model_list[5](input_target).detach().numpy()

    softmaxed0 = F.softmax(torch.tensor(activations0), dim=1).numpy()
    softmaxed1 = F.softmax(torch.tensor(activations1), dim=1).numpy()
    softmaxed2 = F.softmax(torch.tensor(activations2), dim=1).numpy()
    softmaxed3 = F.softmax(torch.tensor(activations3), dim=1).numpy()
    softmaxed4 = F.softmax(torch.tensor(activations4), dim=1).numpy()
    softmaxed5 = F.softmax(torch.tensor(activations5), dim=1).numpy()
    softmaxed6 = F.softmax(torch.tensor(activations5), dim=1).numpy()


    print("[ Visualize .. ]")
    fig, axes = plt.subplots(7, 3, figsize=(15,10))
    fig.suptitle("Single "+str(target_class)+" to "+str(new_class)+" | input_id: "+str(image_id)+" | "+str(pertube_count)+" Perturbation | without 2 last Dense-Layers")

    axes[0][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[0][0].set_title(str(target_class)+" Input Image")
    axes[0][0].axis('off')
    axes[0][1].imshow(activations0, cmap="cool")
    axes[0][1].set_title("Normal")
    axes[0][1].axis('off')
    axes[0][2].imshow(softmaxed0, cmap="cool")
    axes[0][2].set_title("Strongest Class")
    axes[0][2].axis('off')

    axes[1][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[1][0].set_title("")
    axes[1][0].axis('off')
    axes[1][1].imshow(activations1, cmap="cool")
    axes[1][1].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[0])+"$ | 100 Iters")
    axes[1][1].axis('off')
    axes[1][2].imshow(softmaxed1, cmap="cool")
    axes[1][2].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[0])+"$ | 100 Iters")
    axes[1][2].axis('off')

    axes[2][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[2][0].set_title("")
    axes[2][0].axis('off')
    axes[2][1].imshow(activations2, cmap="cool")
    axes[2][1].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[1])+"$ | 100 Iters")
    axes[2][1].axis('off')
    axes[2][2].imshow(softmaxed2, cmap="cool")
    axes[2][2].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[1])+"$ | 100 Iters")
    axes[2][2].axis('off')

    axes[3][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[3][0].set_title("")
    axes[3][0].axis('off')
    axes[3][1].imshow(activations3, cmap="cool")
    axes[3][1].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[2])+"$ | 100 Iters")
    axes[3][1].axis('off')
    axes[3][2].imshow(softmaxed3, cmap="cool")
    axes[3][2].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[2])+"$ | 100 Iters")
    axes[3][2].axis('off')

    axes[4][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[4][0].set_title("")
    axes[4][0].axis('off')
    axes[4][1].imshow(activations4, cmap="cool")
    axes[4][1].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[3])+"$ | 100 Iters")
    axes[4][1].axis('off')
    axes[4][2].imshow(softmaxed4, cmap="cool")
    axes[4][2].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[3])+"$ | 100 Iters")
    axes[4][2].axis('off')

    axes[5][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[5][0].set_title("")
    axes[5][0].axis('off')
    axes[5][1].imshow(activations5, cmap="cool")
    axes[5][1].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[4])+"$ | 100 Iters")
    axes[5][1].axis('off')
    axes[5][2].imshow(softmaxed5, cmap="cool")
    axes[5][2].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[4])+"$ | 100 Iters")
    axes[5][2].axis('off')

    axes[6][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[6][0].set_title("")
    axes[6][0].axis('off')
    axes[6][1].imshow(activations6, cmap="cool")
    axes[6][1].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[5])+"$ | 100 Iters")
    axes[6][1].axis('off')
    axes[6][2].imshow(softmaxed6, cmap="cool")
    axes[6][2].set_title(str(loss_dict[loss_fn])+"-Loss with $\epsilon="+str(epsilons[5])+"$ | 100 Iters")
    axes[6][2].axis('off')

    #fig.text(.5, .05, "Class 4: Deer, Class 7: Horse", ha='center')
    #plt.show()
    #plt.savefig('./'+ str(save_path) +'/layer_eval_'+ str(model_name) +'.png', dpi=400)
