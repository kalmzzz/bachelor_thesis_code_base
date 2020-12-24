import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from matplotlib import pyplot as plt
from advertorch.attacks import L2PGDAttack
import os
import numpy as np
import time
from datetime import datetime
from tqdm import tqdm
import pkbar
from models import *

class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
BCE, WASSERSTEIN, KLDIV = 0, 1, 2
#device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_date():
    d = datetime.now()
    imgYear = "%04d" % (d.year)
    imgMonth = "%02d" % (d.month)
    imgDate = "%02d" % (d.day)
    imgHour = "%02d" % (d.hour)
    imgMins = "%02d" % (d.minute)
    timestamp = "" + str(imgDate) + "." + str(imgMonth) + "." + str(imgYear) + " " + str(imgHour) + ":" + str(imgMins)
    return timestamp

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
        nn.ReLU(inplace=True)
    )
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net.eval()

    basic_complete = CNN()
    checkpoint_complete = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name))
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

def analyze_layers(EPS, ITERS, target_class, new_class, save_path, model_name, pert_count, loss_fn, device_name, layer_cut, target_id=None):
    '''
    analyzes the whole layer activation of the second last and last layer.
    input: target_class, new_class (to compare and generate adversary example)
    '''
    print("[ Initialize .. ]")
    global device
    device = device_name
    loss_function = ""
    if loss_fn == KLDIV:
        loss_function = "KLDiv"
    if loss_fn == WASSERSTEIN:
        loss_function = "Wasserstein"
    if loss_fn == BCE:
        loss_function = "BCE_WithLogits"

    layer_string = ""
    if layer_cut == 2:
        layer_string = "without 2 last layers"
    if layer_cut == 1:
        layer_string = "without 1 last layers"

    date = get_date()
    model, model_complete = get_model(model_name)
    loader1, loader2 = get_loader()
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

    softmaxed_last = F.softmax(torch.tensor(activations_last), dim=1).numpy()
    softmaxed_last2 = F.softmax(torch.tensor(activations_last2), dim=1).numpy()
    softmaxed_advs_last = F.softmax(torch.tensor(activations_advs_last), dim=1).numpy()


    print("[ Visualize .. ]")
    fig, axes = plt.subplots(2, 4, figsize=(13,7))
    fig.suptitle("model activations | input_id: "+str(target_id)+" | $\epsilon= "+str(EPS)+"$ | iters="+str(ITERS)+" | "+str(pert_count)+" Perturbation | "+str(loss_function)+" | "+str(layer_string)+" | " + str(date))
    axes[0][0].imshow(np.moveaxis(input_target.cpu().squeeze().numpy(), 0, -1))
    axes[0][0].set_title("Input Image")
    axes[0][0].axis('off')
    axes[0][1].imshow(activations, cmap="cool")
    axes[0][1].set_title("Activations Penultimate Layer")
    axes[0][1].axis('off')
    axes[0][2].imshow(activations_last, cmap="cool")
    axes[0][2].set_title("Activations Output Layer")
    axes[0][2].get_yaxis().set_visible(False)
    axes[0][2].get_xaxis().set_ticks(np.arange(10))
    axes[0][3].imshow(softmaxed_last, cmap="cool")
    axes[0][3].set_title("Strongest Class")
    axes[0][3].get_yaxis().set_visible(False)
    axes[0][3].get_xaxis().set_ticks(np.arange(10))


    # axes[1][0].imshow(np.moveaxis(advs_img.cpu().squeeze().numpy(), 0, -1))
    # axes[1][0].set_title("Advs. Input " + '$\epsilon='+str(EPS)+'$' + " iters="+str(ITERS))
    # axes[1][0].axis('off')
    # axes[1][1].imshow(activations_advs, cmap="cool")
    # axes[1][1].set_title("Activations Second Last Layer")
    # axes[1][1].axis('off')
    # axes[1][2].imshow(activations_advs_last, cmap="cool")
    # axes[1][2].set_title("Activations Last Layer")
    # axes[1][2].axis('off')
    # axes[1][3].imshow(softmaxed_advs_last, cmap="cool")
    # axes[1][3].set_title("")
    # axes[1][3].axis('off')

    axes[1][0].imshow(np.moveaxis(input_new_class.cpu().squeeze().numpy(), 0, -1))
    axes[1][0].set_title("Example " + str(class_dict[new_class]) + " Input Image")
    axes[1][0].axis('off')
    axes[1][1].imshow(activations2, cmap="cool")
    axes[1][1].set_title("Activations Penultimate Layer")
    axes[1][1].axis('off')
    axes[1][2].imshow(activations_last2, cmap="cool")
    axes[1][2].set_title("Activations Output Layer")
    axes[1][2].get_yaxis().set_visible(False)
    axes[1][2].get_xaxis().set_ticks(np.arange(10))
    axes[1][3].imshow(softmaxed_last2, cmap="cool")
    axes[1][3].set_title("Strongest Class")
    axes[1][3].get_yaxis().set_visible(False)
    axes[1][3].get_xaxis().set_ticks(np.arange(10))
    #plt.show()
    plt.savefig('./'+ str(save_path) +'/layer_eval_'+ str(model_name) +'.png', dpi=400)

    del model
    del model_complete
    del loader1
    del loader2
    del adversary
