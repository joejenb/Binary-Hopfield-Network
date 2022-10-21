import torch
import torch.nn as nn

class BinaryHopfield(nn.Module):
    #init ialize network variables and memory
    def __init__(self, num_minima, num_units):

        self.num_units = num_units
        self.num_minima = num_minima
        self.weights = torch.rand(self.num_units, self.num_units)
        self.units = torch.zeros(1, self.num_units)

    def forward(self, X):
        #Start with batch size of 1
        n = X.size()[0]
        X = X.reshape(-1, self.num_units, 1)

        if self.train():
            #Need to look at how feed in batch of images to learn
            self.weights = torch.sum(X * X.t(), dim=0, keepdim=True) / self.num_minima
            self.weights.fill_diagonal(0)
        else:
            pre_thresh = self.weights * X
            output = torch.where(pre_thresh >= 0, 1, -1)

            energy = -0.5 * self.units.T * self.weights * output
            return energy, output

