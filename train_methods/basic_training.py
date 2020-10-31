import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn

import torchvision
import torchvision.transforms as transforms
from advertorch.attacks import L2PGDAttack

import os
import pkbar

from models import *

file_name = 'basic_training'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = getCNN()
net = net.to(device)
cudnn.benchmark = True

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transform_train)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)


def basic_training_main(num_epochs, learning_rate):
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    criterion = nn.CrossEntropyLoss()
    for epoch in range(0, num_epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=num_epochs, width=30, always_stateful=True)
        adjust_learning_rate(optimizer, epoch, learning_rate)
        net.train()
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            benign_outputs = net(inputs)
            loss = criterion(benign_outputs, targets)
            loss.backward()

            optimizer.step()
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
