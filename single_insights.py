import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import torch
import torchvision
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF

from torchvision import models

from captum.attr import IntegratedGradients
from captum.attr import Saliency
from captum.attr import DeepLift
from captum.attr import NoiseTunnel
from captum.attr import visualization as viz

from models import *

trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor()

)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=4, shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor()

)
testloader = torch.utils.data.DataLoader(testset, batch_size=10000, shuffle=False, num_workers=1)

def get_model():
    net_complete = CNN()
    checkpoint = torch.load('./checkpoint/basic_training_non_robust')
    net_complete.load_state_dict(checkpoint['net'])
    net_complete.eval()

    net_normal = CNN()
    checkpoint = torch.load('./checkpoint/basic_training_with_softmax')
    net_normal.load_state_dict(checkpoint['net'])
    net_normal.eval()
    return net_complete, net_normal


if __name__ == "__main__":
    net, net_normal = get_model()
    dataiter = iter(testloader)
    images, labels = dataiter.next()

    ind = 9035

    input = images[ind].unsqueeze(0)
    input.requires_grad = True

    saliency = Saliency(net)
    grads = saliency.attribute(input, target=labels[ind].item())
    grads = np.transpose(grads.squeeze().cpu().detach().numpy(), (1, 2, 0))

    input = input.squeeze().detach().numpy()
    input = np.moveaxis(input, 0, -1)
    features = np.where(grads > 0.3, input, 0.)

    print("normal prediction: ", net(torch.from_numpy(np.moveaxis(features, -1, 0)).unsqueeze(0)).max(1))
    print("non-robust prediction: ", net(torch.from_numpy(np.moveaxis(features, -1, 0)).unsqueeze(0)).max(1))

    plt.imshow(features)
    plt.show()


    original_image = np.transpose((images[ind].cpu().detach().numpy()), (1, 2, 0))

    _ = viz.visualize_image_attr(None, original_image, method="original_image", title="Original Image")

    _ = viz.visualize_image_attr(grads, original_image, method="blended_heat_map", sign="all", show_colorbar=True, title="Overlayed Gradient Magnitudes")
