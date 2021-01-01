import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
#from sklearn.manifold import TSNE
from MulticoreTSNE import MulticoreTSNE as TSNE
import os
import numpy as np
from tqdm import tqdm
from models import *
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}

cudnn.benchmark = True

def get_model():
    basic_net = CNN()
    checkpoint = torch.load('./checkpoint/basic_training')
    basic_net.fc_layer = nn.Sequential(
        nn.Dropout(p=0.1),
        nn.Linear(4096, 1024),
        nn.ReLU(inplace=True),
        nn.Linear(1024, 512),
        nn.ReLU(inplace=True)
    )
    basic_net.load_state_dict(checkpoint['net'], strict=False)
    basic_net = basic_net.to(device)
    basic_net.eval()
    return basic_net

# ---------------------------------------------------
def scale_to_01_range(x):
    value_range = (np.max(x) - np.min(x))
    starts_from_zero = x - np.min(x)
    return starts_from_zero / value_range


if __name__ == "__main__":
    print("[ Initialize.. ]")
    model = get_model()
    train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)

    features = None
    for idx, (input, target) in tqdm(enumerate(train_loader), desc="Running Model Inference"):
        input = input.to(device)

        with torch.no_grad():
            output = model.forward(input)

        current_features = output.cpu().numpy()
        if features is not None:
            features = np.concatenate((features, current_features))
        else:
            features = current_features

    print("[ Running TSNE ]")
    tsne = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features)
    tx = tsne[:, 0]
    ty = tsne[:, 1]

    tx = scale_to_01_range(tx)
    ty = scale_to_01_range(ty)

    fig = plt.figure()
    fig.suptitle("t_SNE | basic_training | CIFAR10 ")
    ax = fig.add_subplot(111)

    print("[ Visualize.. ]")
    classes = [0,1,2,3,4,5,6,7,8,9]
    colors = ['red', 'green', 'blue', 'lime', 'cornflowerblue', 'magenta', 'gray', 'teal', 'olive', 'peru']
    for single_class in classes:
        labels = train_dataset.targets
        indices = [i for i, l in enumerate(labels) if l == single_class]

        current_tx = np.take(tx, indices)
        current_ty = np.take(ty, indices)

        ax.scatter(current_tx, current_ty, c=colors[single_class], label=class_dict[single_class])

    ax.axis('off')
    ax.legend(loc='best')
    plt.rcParams.update({'font.size': 16})
    plt.show()
