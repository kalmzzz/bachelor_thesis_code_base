import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
from advertorch.attacks import L2PGDAttack, LinfPGDAttack
import torchvision.transforms as transforms
import os
import pkbar
from models import *

class LinfPGDAttack(object):
    def __init__(self, model, epsilon, alpha, k):
        self.model = model
        self.epsilon = epsilon
        self.k = k
        self.alpha = alpha

    def perturb(self, x_natural, y):
        x = x_natural.detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.k):
            x.requires_grad_()
            with torch.enable_grad():
                logits = self.model(x)
                loss = F.cross_entropy(logits, y)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.alpha * torch.sign(grad.detach())
            x = torch.min(torch.max(x, x_natural - self.epsilon), x_natural + self.epsilon)
            x = torch.clamp(x, 0, 1)
        return x

file_name = 'pgd_adversarial_training'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = CNN()
net = net.to(device)
net = torch.nn.DataParallel(net)
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
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=8)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=8)


def advs_training_main(epochs, learning_rate, epsilon, iterations, alpha):
    adversary = LinfPGDAttack(net, epsilon, alpha, iterations)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    for epoch in range(0, 100):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=epochs, width=30, always_stateful=True)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

            adv = adversary.perturb(inputs, targets)
            adv_outputs = net(adv)
            loss = criterion(adv_outputs, targets)
            loss.backward()

            optimizer.step()
            train_loss += loss.item()
            _, predicted = adv_outputs.max(1)

            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            kbar.update(batch_idx, values=[("loss", loss.item()), ("acc", 100. * correct / total)])
        print()
    save_model()

def save_model():
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    torch.save(state, './checkpoint/' + file_name)

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 50:
        lr /= 10
    if epoch >= 75:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
