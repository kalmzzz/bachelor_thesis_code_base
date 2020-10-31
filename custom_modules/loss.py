import torch
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class MSE_Loss(torch.nn.Module):

    def __init__(self):
        super(MSE_Loss, self).__init__()

    def forward(self, y, y_hat):
        return torch.mean(torch.square(torch.sub(y, y_hat))).to(device)
