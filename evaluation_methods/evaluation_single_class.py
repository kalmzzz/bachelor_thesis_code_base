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
import matplotlib as mpl


device = 'cuda' if torch.cuda.is_available() else 'cpu'

transform_test = transforms.Compose([
    transforms.ToTensor(),
])

test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=False, transform=transform_test)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=40, shuffle=False, num_workers=6)
AIRPLANE, AUTO, BIRD, CAT, DEER, DOG, FROG, HORSE, SHIP, TRUCK = 0, 1, 2, 3, 4, 5, 6, 7, 8, 9


def test(target_class, model_path, new_class=None):
    net = getCNN()
    net = net.to(device)
    checkpoint = torch.load('./checkpoint/' + model_path)
    net.load_state_dict(checkpoint['net'])
    cudnn.benchmark = True


    print('\n[ Evaluation Start ]')
    net.eval()

    total = 0
    airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0

    adversary  = L2PGDAttack(net, loss_fn=nn.CrossEntropyLoss(), eps=1.0, nb_iter=12, eps_iter=0.2, rand_init=True, clip_min=0.0, clip_max=1.0, targeted=True)

    for batch_idx, (inputs, targets) in enumerate(tqdm(test_loader)):
        inputs, targets = inputs.to(device), targets.to(device)

        target_ids = torch.where(targets == target_class)[0]
        for id in target_ids:

            if new_class is not None:
                inputs[id] = adversary.perturb(torch.unsqueeze(inputs[id], 0), torch.LongTensor([new_class]).to(device))

            output = net(torch.unsqueeze(inputs[id], 0))
            _, predicted = output.max(1)

            if predicted == AIRPLANE:
                airplane += 1
            if predicted == AUTO:
                auto += 1
            if predicted == BIRD:
                bird += 1
            if predicted == CAT:
                cat += 1
            if predicted == DEER:
                deer += 1
            if predicted == DOG:
                dog += 1
            if predicted == FROG:
                frog += 1
            if predicted == HORSE:
                horse += 1
            if predicted == SHIP:
                ship += 1
            if predicted == TRUCK:
                truck += 1
            total += 1

    benign = [airplane, auto, bird, cat, deer, dog, frog, horse, ship, truck]
    for id_i, item in enumerate(benign):
        benign[id_i] = (item / total)*100.
    del net
    return benign


def evaluate_single_class(model_path):
    benign  = test(target_class=5, model_path = model_path)
    benign2 = test(target_class=3, model_path = model_path)
    benign3 = test(target_class=3, model_path = model_path, new_class=5)

    # benign4 = test(target_class=5, model_path = 'basic_training_single_cat_to_dog')
    # benign5 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog')
    # benign6 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog', new_class=5)
    #
    # benign7 = test(target_class=5, model_path = 'basic_training_single_cat_to_dog')
    # benign8 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog')
    # benign9 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog', new_class=5)
    #
    # benign10 = test(target_class=5, model_path = 'basic_training_single_cat_to_dog')
    # benign11 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog')
    # benign12 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog', new_class=5)
    #
    # benign13 = test(target_class=5, model_path = 'basic_training_single_cat_to_dog')
    # benign14 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog')
    # benign15 = test(target_class=3, model_path = 'basic_training_single_cat_to_dog', new_class=5)

    mpl.style.use('seaborn-deep')
    labels = ["airplane", "auto", "bird", "cat", "deer", "dog", "frog", "horse", "ship", "truck"]
    colors = ["cornflowerblue", "cornflowerblue", "cornflowerblue", "orangered", "cornflowerblue", "orangered", "cornflowerblue", "cornflowerblue", "cornflowerblue", "cornflowerblue"]
    x = np.arange(len(labels))
    y = np.arange(100)
    width = 0.3

    # fig, ((ax, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12), (ax13, ax14, ax15)) = plt.subplots(5, 3, figsize=(15,20))
    fig, (ax, ax2, ax3) = plt.subplots(1, 3, figsize=(15,20))
    fig.suptitle("Single Cat To Dog")

# --------------------------------------------------------------------------------------------------------------------------
    std_rect = ax.bar(x - width/2, benign, width, label='Acc.')
    ax.set_ylabel('Accuracy')
    ax.set_title('input: dog')
    ax.set_xticks(x)
    ax.set_yticks(y)
    ax.set_xticklabels(labels)
    ax.set_yticklabels([])
    ax.legend()
    for rect in std_rect:
        height = rect.get_height()
        ax.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    std_rect2 = ax2.bar(x - width/2, benign2, width, label='Acc.')
    ax2.set_ylabel('')
    ax2.set_title('input: cat')
    ax2.set_xticks(x)
    ax2.set_yticks(y)
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels([])
    ax2.legend()
    for rect in std_rect2:
        height = rect.get_height()
        ax2.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')

    std_rect3 = ax3.bar(x - width/2, benign3, width, label='Acc.')
    ax3.set_ylabel('')
    ax3.set_title('input: advs. dog')
    ax3.set_xticks(x)
    ax3.set_yticks(y)
    ax3.set_xticklabels(labels)
    ax3.set_yticklabels([])
    ax3.legend()
    for rect in std_rect3:
        height = rect.get_height()
        ax3.annotate('{}'.format(np.round(height, 1)),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha='center', va='bottom')
# --------------------------------------------------------------------------------------------------------------------------
#     std_rect4 = ax4.bar(x - width/2, benign4, width, label='Acc.')
#     ax4.set_ylabel('75% pertubation')
#     ax4.set_title('')
#     ax4.set_xticks(x)
#     ax4.set_yticks(y)
#     ax4.set_xticklabels(labels)
#     ax4.set_yticklabels([])
#     ax4.legend()
#     for rect in std_rect4:
#         height = rect.get_height()
#         ax4.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect5 = ax5.bar(x - width/2, benign5, width, label='Acc.')
#     ax5.set_ylabel('')
#     ax5.set_title('')
#     ax5.set_xticks(x)
#     ax5.set_yticks(y)
#     ax5.set_xticklabels(labels)
#     ax5.set_yticklabels([])
#     ax5.legend()
#     for rect in std_rect5:
#         height = rect.get_height()
#         ax5.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect6 = ax6.bar(x - width/2, benign6, width, label='Acc.')
#     ax6.set_ylabel('')
#     ax6.set_title('')
#     ax6.set_xticks(x)
#     ax6.set_yticks(y)
#     ax6.set_xticklabels(labels)
#     ax6.set_yticklabels([])
#     ax6.legend()
#     for rect in std_rect6:
#         height = rect.get_height()
#         ax6.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# # --------------------------------------------------------------------------------------------------------------------------
#     std_rect7 = ax7.bar(x - width/2, benign7, width, label='Acc.')
#     ax7.set_ylabel('50% pertubation')
#     ax7.set_title('')
#     ax7.set_xticks(x)
#     ax7.set_yticks(y)
#     ax7.set_xticklabels(labels)
#     ax7.set_yticklabels([])
#     ax7.legend()
#     for rect in std_rect7:
#         height = rect.get_height()
#         ax7.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect8 = ax8.bar(x - width/2, benign8, width, label='Acc.')
#     ax8.set_ylabel('')
#     ax8.set_title('')
#     ax8.set_xticks(x)
#     ax8.set_yticks(y)
#     ax8.set_xticklabels(labels)
#     ax8.set_yticklabels([])
#     ax8.legend()
#     for rect in std_rect8:
#         height = rect.get_height()
#         ax8.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect9 = ax9.bar(x - width/2, benign9, width, label='Acc.')
#     ax9.set_ylabel('')
#     ax9.set_title('')
#     ax9.set_xticks(x)
#     ax9.set_yticks(y)
#     ax9.set_xticklabels(labels)
#     ax9.set_yticklabels([])
#     ax9.legend()
#     for rect in std_rect9:
#         height = rect.get_height()
#         ax9.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# # --------------------------------------------------------------------------------------------------------------------------
#     std_rect10 = ax10.bar(x - width/2, benign10, width, label='Acc.')
#     ax10.set_ylabel('25% pertubation')
#     ax10.set_title('')
#     ax10.set_xticks(x)
#     ax10.set_yticks(y)
#     ax10.set_xticklabels(labels)
#     ax10.set_yticklabels([])
#     ax10.legend()
#     for rect in std_rect10:
#         height = rect.get_height()
#         ax10.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect11 = ax11.bar(x - width/2, benign11, width, label='Acc.')
#     ax11.set_ylabel('')
#     ax11.set_title('')
#     ax11.set_xticks(x)
#     ax11.set_yticks(y)
#     ax11.set_xticklabels(labels)
#     ax11.set_yticklabels([])
#     ax11.legend()
#     for rect in std_rect11:
#         height = rect.get_height()
#         ax11.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect12 = ax12.bar(x - width/2, benign12, width, label='Acc.')
#     ax12.set_ylabel('')
#     ax12.set_title('')
#     ax12.set_xticks(x)
#     ax12.set_yticks(y)
#     ax12.set_xticklabels(labels)
#     ax12.set_yticklabels([])
#     ax12.legend()
#     for rect in std_rect12:
#         height = rect.get_height()
#         ax12.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# # --------------------------------------------------------------------------------------------------------------------------
#     std_rect13 = ax13.bar(x - width/2, benign13, width, label='Acc.')
#     ax13.set_ylabel('0% pertubation')
#     ax13.set_title('')
#     ax13.set_xticks(x)
#     ax13.set_yticks(y)
#     ax13.set_xticklabels(labels)
#     ax13.set_yticklabels([])
#     ax13.legend()
#     for rect in std_rect13:
#         height = rect.get_height()
#         ax13.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect14 = ax14.bar(x - width/2, benign14, width, label='Acc.')
#     ax14.set_ylabel('')
#     ax14.set_title('')
#     ax14.set_xticks(x)
#     ax14.set_yticks(y)
#     ax14.set_xticklabels(labels)
#     ax14.set_yticklabels([])
#     ax14.legend()
#     for rect in std_rect14:
#         height = rect.get_height()
#         ax14.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#     std_rect15 = ax15.bar(x - width/2, benign15, width, label='Acc.')
#     ax15.set_ylabel('')
#     ax15.set_title('')
#     ax15.set_xticks(x)
#     ax15.set_yticks(y)
#     ax15.set_xticklabels(labels)
#     ax15.set_yticklabels([])
#     ax15.legend()
#     for rect in std_rect15:
#         height = rect.get_height()
#         ax15.annotate('{}'.format(np.round(height, 1)),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),
#                     textcoords="offset points",
#                     ha='center', va='bottom')
# --------------------------------------------------------------------------------------------------------------------------
    plt.savefig('results/evaluation_'+ str(model_path) +'.png', dpi=400)
    plt.show()
