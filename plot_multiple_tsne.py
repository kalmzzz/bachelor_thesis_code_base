import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from sklearn.manifold import TSNE
import os
import numpy as np
from tqdm import tqdm
from models import *
import matplotlib.pyplot as plt

device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_dict = {0:"Airplane", 1:"Auto", 2:"Bird", 3:"Cat", 4:"Deer", 5:"Dog", 6:"Frog", 7:"Horse", 8:"Ship", 9:"Truck"}
AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9

def get_model(path):
    basic_net = CNN()
    checkpoint = torch.load('./checkpoint/'+str(path))
    basic_net.fc_layer = nn.Sequential(
        nn.Identity()
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
    target_class = DEER
    new_class = HORSE
    data_suffix = "basic_training_single_deer_to_horse_kldiv_no_softmax"
    target_image_id = 1293
    poison_ids = None


    print("[ Initialize.. ]")
    model0 = get_model(data_suffix+"_0")
    model1 = get_model(data_suffix+"_25")
    model2 = get_model(data_suffix+"_75")
    model3 = get_model(data_suffix+"_100")

    data_path = "madry_data/release_datasets/perturbed_CIFAR/"
    train_data = ch.load(os.path.join(data_path, f"CIFAR_ims_"+str(data_suffix)))
    train_labels = ch.load(os.path.join(data_path, f"CIFAR_lab_"+str(data_suffix)))
    train_dataset = TensorDataset(train_data, train_labels, transform=transforms.ToTensor())
    test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transforms.ToTensor())
    train_dataset = ch.cat(train_dataset, test_dataset[target_image_id])
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=128, shuffle=False, num_workers=4)

    features0, features1, features2, features3 = None, None, None, None
    for idx, (input, target) in tqdm(enumerate(train_loader), desc="Running Model Inference"):
        input = input.to(device)
        if target == target_class
            with torch.no_grad():
                output0 = model0.forward(input)
                output1 = model1.forward(input)
                output2 = model2.forward(input)
                output3 = model3.forward(input)

            current_features0 = output0.cpu().numpy()
            current_features1 = output1.cpu().numpy()
            current_features2 = output2.cpu().numpy()
            current_features3 = output3.cpu().numpy()
            if features is not None:
                features0 = np.concatenate((features0, current_features0))
                features1 = np.concatenate((features1, current_features1))
                features2 = np.concatenate((features2, current_features2))
                features3 = np.concatenate((features3, current_features3))
            else:
                features0 = current_features0
                features1 = current_features1
                features2 = current_features2
                features3 = current_features3


    print("[ Running TSNE ]")
    tsne0 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features0)
    tsne1 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features1)
    tsne2 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features2)
    tsne3 = TSNE(n_components=2, verbose=1, n_jobs=-1).fit_transform(features3)

    tx0, ty0, tx1, ty1, tx2, ty2 ,tx3, ty3 = tsne0[:, 0], tsne0[:, 1], tsne1[:, 0], tsne1[:, 1], tsne2[:, 0], tsne2[:, 1], tsne3[:, 0], tsne3[:, 1]

    tx0, ty0 = scale_to_01_range(tx0), scale_to_01_range(ty0)
    tx1, ty1 = scale_to_01_range(tx1), scale_to_01_range(ty1)
    tx2, ty2 = scale_to_01_range(tx2), scale_to_01_range(ty2)
    tx3, ty3 = scale_to_01_range(tx3), scale_to_01_range(ty3)

    fig = plt.figure()
    fig.suptitle("t_SNE | basic_training_with_softmax | CIFAR10 | Testset")

    ax0 = fig.add_subplot(411)
    ax2 = fig.add_subplot(412)
    ax3 = fig.add_subplot(413)
    ax4 = fig.add_subplot(414)

    print("[ Visualize.. ]")
    classes = [target_class, new_class]
    colors = ['red', 'green']
    for single_class in classes:
        labels = train_dataset.targets
        indices = [i for i, l in enumerate(labels) if l == single_class]

        current_tx0, current_ty0 = np.take(tx0, indices), np.take(ty0, indices)
        current_tx1, current_ty1 = np.take(tx1, indices), np.take(ty1, indices)
        current_tx2, current_ty2 = np.take(tx2, indices), np.take(ty2, indices)
        current_tx3, current_ty3 = np.take(tx3, indices), np.take(ty3, indices)

        ax0.scatter(current_tx0, current_ty0, c=colors[single_class], label=class_dict[single_class])
        ax1.scatter(current_tx1, current_ty1, c=colors[single_class], label=class_dict[single_class])
        ax2.scatter(current_tx2, current_ty2, c=colors[single_class], label=class_dict[single_class])
        ax3.scatter(current_tx3, current_ty3, c=colors[single_class], label=class_dict[single_class])

    ax.legend(loc='best')
    plt.show()
