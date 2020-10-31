import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch as ch
from torch.utils.data import Dataset
import torchvision
import torchvision.transforms as transforms
import os
import pkbar
from models import *
from advertorch.attacks import L2PGDAttack


file_name = 'basic_training_with_non_robust_dataset'
device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = getCNN()
net = net.to(device)
cudnn.benchmark = True


transform_train = transforms.Compose([
    transforms.ToTensor(),
])
transform_test = transforms.Compose([
    transforms.ToTensor(),
])
class TensorDataset(Dataset):
    def __init__(self, *tensors, transform=None):
        assert all(tensors[0].size(0) == tensor.size(0) for tensor in tensors)
        self.tensors = tensors
        self.transform = transform

    def __getitem__(self, index):
        im, targ = tuple(tensor[index] for tensor in self.tensors)
        if self.transform:
            real_transform = transforms.Compose([
                transforms.ToPILImage(),
                self.transform
            ])
            im = real_transform(im)
        return im, targ

    def __len__(self):
        return self.tensors[0].size(0)

def get_loaders():
    data_path = "madry_data/release_datasets/d_non_robust_CIFAR/"
    train_data = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_ims")))
    train_labels = ch.cat(ch.load(os.path.join(data_path, f"CIFAR_lab")))
    train_dataset = TensorDataset(train_data, train_labels, transform=transform_train)
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=10)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=100, shuffle=False, num_workers=10)
    return train_loader, test_loader


def non_robust_training_main(epochs, learning_rate):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_loader, test_loader = get_loaders()
    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=100, width=20, always_stateful=True)
        net.train()
        train_loss = 0
        correct = 0
        total = 0
        for batch_idx, (inputs, targets) in enumerate(train_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()

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
