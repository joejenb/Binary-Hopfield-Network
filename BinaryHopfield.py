import torch
import torch.nn as nn

class BinaryHopfield(nn.Module):
    #init ialize network variables and memory
    def __init__(self, num_minima, num_units):
        super(BinaryHopfield, self).__init__()

        self.num_units = num_units
        self.num_minima = num_minima
        self.weights = torch.rand(self.num_units, self.num_units)
        self.units = torch.zeros(1, self.num_units)

    def forward(self, X):
        #Start with batch size of 1
        n = X.size()[0]
        X_flat = X.reshape(self.num_units, 1).float()

        self.units = X_flat

        if self.training:
            #Need to look at how feed in batch of images to learn
            #self.weights = torch.sum(self.units.matmul(self.units.T), dim=0, keepdim=True) / self.num_minima
            self.weights = self.units.matmul(self.units.T) / self.num_minima
            self.weights.fill_diagonal_(0)
        else:
            self.units = self.weights.matmul(self.units)
            self.units = torch.where(self.units >= 0, 1, -1).float()

            energy = -0.5 * self.units.T.matmul(self.weights.matmul(self.units))
            return energy, self.units.reshape(X.size())
