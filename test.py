import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from advertorch.attacks import L2PGDAttack
from scipy.stats import wasserstein_distance
from matplotlib import pyplot as plt
from geomloss import SamplesLoss
import os
from math import *
#os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
import numpy as np
import time
import pkbar
from models import *
from custom_modules import Wasserstein_Loss, Wasserstein_Loss2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

cudnn.benchmark = True

def get_model():
    basic_net = CNN()
    checkpoint = torch.load('./model_saves/basic_training/basic_training')
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net.eval()
    basic_net = basic_net.to(device)
    return basic_net

# ---------------------------------------------------
if __name__ == "__main__":
    print("[ Initialize.. ]")
    model = get_model()
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=4)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)

    adversary0 = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(), eps=2.0, nb_iter=100, eps_iter=(2.0/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    adversary1 = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(), eps=1.0, nb_iter=100, eps_iter=(1.0/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    adversary2 = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(), eps=0.75, nb_iter=100, eps_iter=(0.75/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    adversary3 = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(), eps=0.5, nb_iter=100, eps_iter=(0.5/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    adversary4 = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=(0.25/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    adversary5 = L2PGDAttack(model, loss_fn=nn.CrossEntropyLoss(), eps=0.1, nb_iter=100, eps_iter=(0.1/10.), rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)


    original = None
    for idx, (input, target) in enumerate(test_loader):
        if idx == 9035:
            original = input
            break

    advs0 = adversary0.perturb(original.to(device), torch.LongTensor([7]).to(device))
    advs1 = adversary1.perturb(original.to(device), torch.LongTensor([7]).to(device))
    advs2 = adversary2.perturb(original.to(device), torch.LongTensor([7]).to(device))
    advs3 = adversary3.perturb(original.to(device), torch.LongTensor([7]).to(device))
    advs4 = adversary4.perturb(original.to(device), torch.LongTensor([7]).to(device))
    advs5 = adversary5.perturb(original.to(device), torch.LongTensor([7]).to(device))

    fig, axes = plt.subplots(2, 4, figsize=(20,5))
    plt.rcParams.update({'font.size': 16})
    #fig.suptitle("model activations | input_id: "+str(target_id)+" | $\epsilon= "+str(EPS)+"$ | iters="+str(ITERS)+" | "+str(pert_count)+" Perturbation | "+str(loss_function)+" | "+str(layer_string)+" | " + str(date))
    axes[0][0].imshow(np.moveaxis(original.cpu().squeeze().numpy(), 0, -1))
    axes[0][0].set_title("Original Image")
    axes[0][0].axis('off')
    axes[0][1].imshow(np.moveaxis(advs0.cpu().squeeze().numpy(), 0, -1))
    axes[0][1].set_title('$\epsilon=2.0$')
    axes[0][1].axis('off')
    axes[0][2].imshow(np.moveaxis(advs1.cpu().squeeze().numpy(), 0, -1))
    axes[0][2].set_title('$\epsilon=1.0$')
    axes[0][2].axis('off')
    axes[0][3].imshow(np.moveaxis(advs1.cpu().squeeze().numpy(), 0, -1))
    axes[0][3].set_title('')
    axes[0][3].axis('off')
    axes[1][0].imshow(np.moveaxis(advs2.cpu().squeeze().numpy(), 0, -1))
    axes[1][0].set_title('$\epsilon=0.75$')
    axes[1][0].axis('off')
    axes[1][1].imshow(np.moveaxis(advs3.cpu().squeeze().numpy(), 0, -1))
    axes[1][1].set_title('$\epsilon=0.5$')
    axes[1][1].axis('off')
    axes[1][2].imshow(np.moveaxis(advs4.cpu().squeeze().numpy(), 0, -1))
    axes[1][2].set_title('$\epsilon=0.25$')
    axes[1][2].axis('off')
    axes[1][3].imshow(np.moveaxis(advs5.cpu().squeeze().numpy(), 0, -1))
    axes[1][3].set_title('$\epsilon=0.1$')
    axes[1][3].axis('off')

    plt.show()
