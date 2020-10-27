import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision
import torchvision.transforms as transforms
from models import *
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from advertorch.attacks import LinfPGDAttack, L2PGDAttack
import pkbar

model1_path = 'basic_training'
model2_path = 'pgd_adversarial_training'
model3_path = 'basic_training_with_robust_dataset'
model4_path = 'basic_training_with_non_robust_dataset'
model5_path = 'basic_training_with_drand_dataset'
model6_path = 'basic_training_with_ddet_dataset'

device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=6)

net = CNN()
net = net.to(device)
net = torch.nn.DataParallel(net)
checkpoint = torch.load('./checkpoint/' + model1_path)
net.load_state_dict(checkpoint['net'])

net2 = CNN()
net2 = net2.to(device)
net2 = torch.nn.DataParallel(net2)
checkpoint2 = torch.load('./checkpoint/' + model2_path)
net2.load_state_dict(checkpoint2['net'])

net3 = CNN()
net3 = net3.to(device)
net3 = torch.nn.DataParallel(net3)
checkpoint3 = torch.load('./checkpoint/' + model3_path)
net3.load_state_dict(checkpoint3['net'])

net4 = CNN()
net4 = net4.to(device)
net4 = torch.nn.DataParallel(net4)
checkpoint4 = torch.load('./checkpoint/' + model4_path)
net4.load_state_dict(checkpoint4['net'])

net5 = CNN()
net5 = net5.to(device)
net5 = torch.nn.DataParallel(net5)
checkpoint5 = torch.load('./checkpoint/' + model5_path)
net5.load_state_dict(checkpoint5['net'])

net6 = CNN()
net6 = net6.to(device)
net6 = torch.nn.DataParallel(net6)
checkpoint6 = torch.load('./checkpoint/' + model6_path)
net6.load_state_dict(checkpoint6['net'])

cudnn.benchmark = True

#adversary = LinfPGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.0314, nb_iter=7, eps_iter=0.00784, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary  = L2PGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary2 = L2PGDAttack(net2, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary3 = L2PGDAttack(net3, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary4 = L2PGDAttack(net4, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary5 = L2PGDAttack(net5, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)
adversary6 = L2PGDAttack(net6, loss_fn=nn.CrossEntropyLoss(), eps=0.25, nb_iter=100, eps_iter=0.01, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=False)

criterion = nn.CrossEntropyLoss()

def test():
    print('\n[ Evaluation Start ]')
    net.eval()
    net2.eval()
    net3.eval()
    net4.eval()
    net5.eval()
    net6.eval()

    net_benign_correct = 0
    net_adv_correct = 0
    net2_benign_correct = 0
    net2_adv_correct = 0
    net3_benign_correct = 0
    net3_adv_correct = 0
    net4_benign_correct = 0
    net4_adv_correct = 0
    net5_benign_correct = 0
    net5_adv_correct = 0
    net6_benign_correct = 0
    net6_adv_correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(device), targets.to(device)
        total += targets.size(0)
        adv = adversary.perturb(inputs, targets)
        adv2 = adversary2.perturb(inputs, targets)
        adv3 = adversary3.perturb(inputs, targets)
        adv4 = adversary4.perturb(inputs, targets)
        adv5 = adversary5.perturb(inputs, targets)
        adv6 = adversary6.perturb(inputs, targets)

# ------------------------- Basic Training on normal CIFAR10 -----------------
        outputs = net(inputs)
        _, predicted = outputs.max(1)
        net_benign_correct += predicted.eq(targets).sum().item()

        adv_outputs = net(adv)
        _, predicted = adv_outputs.max(1)
        net_adv_correct += predicted.eq(targets).sum().item()
# ------------------------- Adversarial Training on normal CIFAR10 -----------------
        outputs2 = net2(inputs)
        _, predicted2 = outputs2.max(1)
        net2_benign_correct += predicted2.eq(targets).sum().item()

        adv_outputs2 = net2(adv2)
        _, predicted2 = adv_outputs2.max(1)
        net2_adv_correct += predicted2.eq(targets).sum().item()
# ------------------------- Basic Training on robust CIFAR10 -----------------
        outputs3 = net3(inputs)
        _, predicted3 = outputs3.max(1)
        net3_benign_correct += predicted3.eq(targets).sum().item()

        adv_outputs3 = net3(adv3)
        _, predicted3 = adv_outputs3.max(1)
        net3_adv_correct += predicted3.eq(targets).sum().item()
# ------------------------- Basic Training on non robust CIFAR10 -----------------
        outputs4 = net4(inputs)
        _, predicted4 = outputs4.max(1)
        net4_benign_correct += predicted4.eq(targets).sum().item()

        adv_outputs4 = net4(adv4)
        _, predicted4 = adv_outputs4.max(1)
        net4_adv_correct += predicted4.eq(targets).sum().item()
# ------------------------- Basic Training on drand CIFAR10 -----------------
        outputs5 = net5(inputs)
        _, predicted5 = outputs5.max(1)
        net5_benign_correct += predicted5.eq(targets).sum().item()

        adv_outputs5 = net5(adv5)
        _, predicted5 = adv_outputs5.max(1)
        net5_adv_correct += predicted5.eq(targets).sum().item()
# ------------------------- Basic Training on ddet CIFAR10 -----------------
        outputs6 = net6(inputs)
        _, predicted6 = outputs6.max(1)
        net6_benign_correct += predicted6.eq(targets).sum().item()

        adv_outputs6 = net6(adv6)
        _, predicted6 = adv_outputs6.max(1)
        net6_adv_correct += predicted6.eq(targets).sum().item()

# ------------------------- Visualizisation -----------------
    net_benign_correct = (net_benign_correct / total)*100.
    net2_benign_correct = (net2_benign_correct / total)*100.
    net3_benign_correct = (net3_benign_correct / total)*100.
    net4_benign_correct = (net4_benign_correct / total)*100.
    net5_benign_correct = (net5_benign_correct / total)*100.
    net6_benign_correct = (net6_benign_correct / total)*100.

    net_adv_correct = (net_adv_correct / total)*100.
    net2_adv_correct = (net2_adv_correct / total)*100.
    net3_adv_correct = (net3_adv_correct / total)*100.
    net4_adv_correct = (net4_adv_correct / total)*100.
    net5_adv_correct = (net5_adv_correct / total)*100.
    net6_adv_correct = (net6_adv_correct / total)*100.

    benign = [net_benign_correct, net2_benign_correct, net3_benign_correct, net4_benign_correct, net5_benign_correct, net6_benign_correct]
    advs = [net_adv_correct, net2_adv_correct, net3_adv_correct, net4_adv_correct, net5_adv_correct, net6_adv_correct]

    labels = ["Natural Model", "Advs Training Model", "Robust Model", "Non-Robust Model", "DRAND Model", "DDET Model"]
    x = np.arange(len(labels))
    width = 0.3
    fig, ax = plt.subplots()
    std_rect = ax.bar(x - width/2, benign, width, label='Std. Acc.')
    advs_rect = ax.bar(x + width/2, advs, width, label='Advs. Acc.')

    ax.set_ylabel('Accuracy in Percent')
    ax.set_title('Evaluation')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.legend()
    for rect in std_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    for rect in advs_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')
    fig.tight_layout()
    plt.show()


if __name__ == "__main__":
    test()
