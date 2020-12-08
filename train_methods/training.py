import torch
import torch.nn as nn
import torch.optim as optim
import torch as ch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import pkbar
from models import *
from custom_modules import TensorDataset
from advertorch.attacks import L2PGDAttack

#device = 'cuda' if torch.cuda.is_available() else 'cpu'
save_epochs = [0, 24, 49, 74, 99]

def save_model(path, file_name, net):
    print("[ Saving Model ]")
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('model_saves'):
        os.mkdir('model_saves')
    if not os.path.isdir('model_saves/'+str(path)):
        os.mkdir('model_saves/'+str(path))
    torch.save(state, './model_saves/'+str(path)+'/' + str(file_name))


def adjust_learning_rate(optimizer, epoch, epochs, learning_rate):
    lr = learning_rate
    if epoch >= np.floor(epochs*0.5):
        lr /= 10
    if epoch >= np.floor(epochs*0.75):
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def get_loaders(data_suffix, batch_size, data_augmentation):
    if data_augmentation:
        data_transform = transforms.Compose([transforms.RandomCrop(32, padding=4), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    else:
        data_transform = transforms.Compose([transforms.ToTensor()])

    if data_suffix is None:
        train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=data_transform)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    else:
        data_path = "datasets/"+str(data_suffix)+"/"
        train_data = ch.load(os.path.join(data_path, f"CIFAR_ims_"+str(data_suffix))).to(device)
        train_labels = ch.load(os.path.join(data_path, f"CIFAR_lab_"+str(data_suffix))).to(device)
        train_dataset = TensorDataset(train_data, train_labels, transform=data_transform)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=0)
    return train_loader


def get_model():
    net = CNN()
    net = net.to(device)
    return net


def train(epochs, learning_rate, output_name, data_suffix, batch_size, device_name, data_augmentation=False):
    print("[ Initialize Training ]")
    global device
    device = device_name
    net = get_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_loader = get_loaders(data_suffix, batch_size, data_augmentation)

    for epoch in range(0, epochs):
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=epochs, width=20, always_stateful=True)
        adjust_learning_rate(optimizer, epoch, epochs, learning_rate)
        net.train()
        correct = 0
        total = 0
        running_loss = 0.0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()

            optimizer.step()
            _, predicted = outputs.max(1)

            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            kbar.update(batch_idx, values=[("loss", running_loss/(batch_idx+1)), ("acc", 100. * correct / total)])
        print()
        if epoch in save_epochs:
            save_model(output_name, output_name+"_"+str(epoch+1), net)
    save_model(output_name, output_name, net)
