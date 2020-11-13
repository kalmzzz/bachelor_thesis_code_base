import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models import *

device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = CNN()
checkpoint = torch.load('./checkpoint/basic_training_with_softmax')
model.load_state_dict(checkpoint['net'])
model = model.to(device)
model.eval()

class KLDivLoss(torch.nn.Module):

    def __init__(self):
        super(KLDivLoss, self).__init__()

    def forward(self, y, y_hat):
        return F.kl_div(F.log_softmax(y, dim=1), F.softmax(y_hat, dim=1), None, None, reduction='sum')


class Complete_Loss(torch.nn.Module):
    #Minimiert Loss zur gewünschten Klasse und maximiert ihn zu allen anderen Klassen
    def __init__(self, exception_class):
        super(Complete_Loss, self).__init__()
        self.exception_class = exception_class #gibt an zu welcher Klasse der Loss sich nicht maximieren soll
        self.loss_weight = 0.01

    def forward(self, y, y_hat, x):
        output = model(x)
        loss = F.kl_div(F.log_softmax(y, dim=1), F.softmax(y_hat, dim=1), None, None, reduction='sum')
        for class_id in range(0,10):
            if class_id is not self.exception_class:
                loss -= F.cross_entropy(output, torch.LongTensor([class_id]).to(device)) * self.loss_weight

        return loss.to(device)

class Wasserstein_Loss(torch.nn.Module):

    def __init__(self):
        super(Wasserstein_Loss, self).__init__()

    def forward(self, tensor_a, tensor_b):
        tensor_a = tensor_a / (torch.sum(tensor_a, dim=-1, keepdim=True) + 1e-14)
        tensor_b = tensor_b / (torch.sum(tensor_b, dim=-1, keepdim=True) + 1e-14)

        cdf_tensor_a = torch.cumsum(tensor_a, dim=-1)
        cdf_tensor_b = torch.cumsum(tensor_b, dim=-1)

        cdf_distance = torch.sum(torch.abs(torch.sub(cdf_tensor_a, cdf_tensor_b)), dim=-1)
        cdf_loss = cdf_distance.mean() * 0.1

        return cdf_loss

class Wasserstein_Loss2(torch.nn.Module):

    def __init__(self):
        super(Wasserstein_Loss2, self).__init__()

    def forward(self, u_values, v_values):
        u_values, v_values = torch.squeeze(u_values), torch.squeeze(v_values)

        with torch.no_grad():
            u_sorter = torch.argsort(u_values)
            v_sorter = torch.argsort(v_values)

            all_values = torch.cat((u_values, v_values))

            all_values, _ = torch.sort(all_values.detach())

            deltas = torch.sub(all_values[1:], all_values[:-1])

            u_cdf_indices = torch.searchsorted(u_values[u_sorter], all_values[:-1], right=True)
            v_cdf_indices = torch.searchsorted(v_values[v_sorter], all_values[:-1], right=True)

            u_cdf = torch.true_divide(u_cdf_indices, u_values.dim())
            v_cdf = torch.true_divide(v_cdf_indices, v_values.dim())


        return torch.sum(torch.mul(torch.abs(torch.sub(u_cdf, v_cdf)), deltas))
