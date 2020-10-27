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


device = 'cuda' if torch.cuda.is_available() else 'cpu'
net = getCNN()
net = net.to(device)
cudnn.benchmark = True


transform_train = transforms.Compose([
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

def get_loaders(pertubation_factor, iter, eps, inf=False):
    data_path = "madry_data/release_datasets/pertubed_CIFAR/"
    if inf:
        train_data = ch.load(os.path.join(data_path, f'CIFAR_ims_Linf_'+str(pertubation_factor)+'pert_'+str(iter)+'iters_'+str(eps)+'eps'))
        train_labels = ch.load(os.path.join(data_path, f'CIFAR_lab_Linf_'+str(pertubation_factor)+'pert_'+str(iter)+'iters_'+str(eps)+'eps'))
    else:
        train_data = ch.load(os.path.join(data_path, f'CIFAR_ims_L2_'+str(pertubation_factor)+'pert_'+str(iter)+'iters_'+str(eps)+'eps'))
        train_labels = ch.load(os.path.join(data_path, f'CIFAR_lab_L2_'+str(pertubation_factor)+'pert_'+str(iter)+'iters_'+str(eps)+'eps'))
    train_dataset = TensorDataset(train_data, train_labels, transform=transform_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6)
    return train_loader


def pertubated_dataset_training_main(epochs, learning_rate, model_name, pertubation_factor, iter, eps, inf=False):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)
    train_loader = get_loaders(pertubation_factor, iter, eps, inf)
    for epoch in range(0, epochs):
        adjust_learning_rate(optimizer, epoch, learning_rate)
        kbar = pkbar.Kbar(target=len(train_loader), epoch=epoch, num_epochs=100, width=40, always_stateful=True)
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
    save_model(model_name, pertubation_factor, iter, eps, inf)

def save_model(model_name, pertubation_factor, iter, eps, inf):
    state = {
        'net': net.state_dict()
    }
    if not os.path.isdir('checkpoint'):
        os.mkdir('checkpoint')
    if inf:
        torch.save(state, './checkpoint/' + model_name + '_Linf_'+str(pertubation_factor)+'pert_'+str(iter)+'iters_'+str(eps)+'eps')
    else:
        torch.save(state, './checkpoint/' + model_name + '_L2_'+str(pertubation_factor)+'pert_'+str(iter)+'iters_'+str(eps)+'eps')

def adjust_learning_rate(optimizer, epoch, learning_rate):
    lr = learning_rate
    if epoch >= 50:
        lr /= 10
    if epoch >= 75:
        lr /= 10
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
