#imports
import torch
import torch.nn as nn
import torch.nn.functional as F

import wandb

import torchvision
from torchvision import transforms

#for visualization
import matplotlib.pyplot as plt

from BinaryHopfield import BinaryHopfield

def get_data_loaders(config):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
        ])

    train_set = torchvision.datasets.MNIST(root="/MNIST/", train=True, download=True, transform=torchvision.transforms.ToTensor())
    val_set = torchvision.datasets.MNIST(root="/MNIST/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    test_set = torchvision.datasets.MNIST(root="/MNIST/", train=False, download=True, transform=torchvision.transforms.ToTensor())
    num_classes = 10

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=config.batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=config.batch_size, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=config.batch_size, shuffle=False)
    
    return train_loader, val_loader, test_loader, num_classes


def train(BHNet, X, device):

    # Learn memory
    BHNet.train()
    BHNet(X)

    return X


def test(BHNet, X, config, device):
    # Recall Memory
    BHNet.eval() 
    fuzzy_X = torch.rand(X.size()).to(device)

    X_recalls = []
    for update in range(config.epochs):
        energy, fuzzy_X = BHNet(fuzzy_X)

        loss = F.mse_loss(fuzzy_X, X)
        X_recalls.append()

        wandb.log({
            "Recall": wandb.Image(fuzzy_X[0]),
            "Reconstruction Loss": loss,
            "Energy": energy
            })


wandb.init(entity="wandb", project="Binary-Hopfield-Network")
wandb.watch_called = False # Re-run the model without restarting the runtime, unnecessary after our next release

# WandB – Config is a variable that holds and saves hyperparameters and inputs
config = wandb.config          # Initialize config
config.batch_size = 1          # input batch size for training (default: 64)
config.test_batch_size = 1    # input batch size for testing (default: 1000)
config.epochs = 50             # number of epochs to train (default: 10)
config.no_cuda = False         # disables CUDA training
config.seed = 42               # random seed (default: 42)
config.log_interval = 1     # how many batches to wait before logging training status



def main():
    # WandB – Initialize a new run
    #config = init_wandb()

    use_cuda = not config.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")

    train_loader, val_loader, test_loader, num_classes = get_data_loaders(config)

    image_size = 28

    BHNet = BinaryHopfield(1, image_size ** 2).to(device)
    wandb.watch(BHNet, log="all")

    X = next(iter(train_loader))[0].to(device)
    X = torch.where(X > 0.08, 1, -1)

    train(BHNet, X, device)
    test(BHNet, X, config, device)

if __name__ == '__main__':
    main()