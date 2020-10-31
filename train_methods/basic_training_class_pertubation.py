import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from advertorch.attacks import L2PGDAttack

import os
import matplotlib.pyplot as plt
import numpy as np
import pkbar

from models import *

file_name = 'basic_training_cat_to_dog_24iters'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = getCNN()
net = net.to(device)

basic_net = getCNN()
basic_net = basic_net.to(device)
checkpoint = torch.load('./checkpoint/basic_training')
basic_net.load_state_dict(checkpoint['net'])
basic_net.eval()
cudnn.benchmark = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=4)


def basic_training_class_perturbation_main(num_epochs, learning_rate, eps, iterations, new_class, target_class):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    adversary  = L2PGDAttack(basic_net, loss_fn=nn.CrossEntropyLoss(), eps=eps, nb_iter=iterations, eps_iter=0.2, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(0, num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=30, always_stateful=True)
        adjust_learning_rate(optimizer, epoch, learning_rate)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            target_ids = torch.where(targets == target_class)[0]
            for id in target_ids:
                inputs[id] = adversary.perturb(torch.unsqueeze(inputs[id], 0), torch.LongTensor([new_class]).to(device))

            benign_outputs = net(inputs)
            loss = criterion(benign_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = benign_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            kbar.update(batch_idx, values=[("loss", loss.item()), ("acc", 100. * correct / total)])
        print()
        save_model()

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 50:
        lr /= 10
    if epoch >= 75:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def save_model():
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)
