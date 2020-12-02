import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from models import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
import pkbar

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def get_loader():
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=2)
    return test_loader

def get_model(model_name):
    net = CNN()
    net = net.to(device)
    checkpoint = torch.load('./model_saves/'+str(model_name)+'/'+str(model_name))
    net.load_state_dict(checkpoint['net'])
    return net


def single_model_evaluation(model_name, save_path):
    '''
    Evaluates a single model using normal and pertubed images of the whole cifar10 test dataset

    '''
    print("[ Initialize ]")
    net = get_model(model_name)
    test_loader = get_loader()
    adversary  = L2PGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

    print('\n[ Evaluation Start ]')
    net.eval()

    net_benign_correct = 0
    net_adv_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = adversary.perturb(inputs, targets)

        outputs = net(inputs)
        _, predicted = outputs.max(1)
        net_benign_correct += predicted.eq(targets).sum().item()

        adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        net_adv_correct += predicted.eq(targets).sum().item()

# ------------------------- Visualizisation -----------------
    net_benign_correct = (net_benign_correct / total)*100.
    net_adv_correct = (net_adv_correct / total)*100.


    benign = [net_benign_correct]
    advs = [net_adv_correct]

    labels = [model_name]
    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots()
    std_rect = ax.bar(x - width/2, benign, width, label='Std. Acc.')
    advs_rect = ax.bar(x + width/2, advs, width, label='Advs. Acc.')

    ax.set_ylabel('Accuracy in Percent')
    ax.set_title('Evaluation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for rect in std_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in advs_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    plt.savefig('./'+ str(save_path) +'/model_acc_'+ str(model_name) +'.png', dpi=400)
    #plt.show()
