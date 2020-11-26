import torch
import torch as ch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
import os
import numpy as np
import matplotlib.pyplot as plt
from custom_modules import TensorDataset

device = 'cuda' if torch.cuda.is_available() else 'cpu'
AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9
cudnn.benchmark = True

if __name__ == "__main__":
    print("[ Initialize.. ]")
    data_path = "madry_data/release_datasets/perturbed_CIFAR/"
    train_data = ch.load(os.path.join(data_path, f"CIFAR_ims_single_deer_to_horse_grads"))
    train_labels = ch.load(os.path.join(data_path, f"CIFAR_lab_single_deer_to_horse_grads"))
    train_dataset = TensorDataset(train_data, train_labels, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=False, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transforms.ToTensor())
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False, num_workers=1)

    for idx, (input, target) in enumerate(testset):
        if target == DEER:
            print(idx)
            plt.imshow(np.moveaxis(input.squeeze().detach().cpu().numpy(), 0, -1))
            plt.show()
