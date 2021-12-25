import torch
from torch._C import _llvm_enabled
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from torch.optim import optimizer
from model import MLPNet, CNN, AEMNIST
from dataloader import get_loader_MNIST
from ekfac import EKFAC
from kfac import KFAC
from temp import test

# Hyper-parameters
lr = 1 # learning rate
batch_size=1

train_loader, test_loader = get_loader_MNIST(batch_size=batch_size)

# Training
def classification_MNIST():
    """toy experiment on MNIST classification"""
    model = CNN()
    preconditioner = KFAC(model, 0.1, update_freq=1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad() # set tensor's grad attribute to 0
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        print(loss)
        preconditioner.step() # g = F^-1 * g
        optimizer.step() # 1. d = A * d; 2. x' = x + alpha * d
    return model

def evaluate(model, criterion, dataloader):
    """evaluate auto-encoder"""
    loss = 0.
    length = 0
    for test_data, label in dataloader:
        out = model(test_data).view(-1, 28, 28)
        loss += criterion(out, test_data)
        length += 1
    return loss / length

def auto_encoder_MNIST():
    """experiment on autoencoder"""
    model = AEMNIST()
    optimizer = optim.SGD(model.parameters(), lr=lr)
    preconditioner = KFAC(model, 0.1, update_freq=1)
    for i, (inputs, _) in enumerate(train_loader):
        optimizer.zero_grad() # set tensor's grad attribute to 0
        outputs = model(inputs).view(-1, 28, 28)
        loss = F.mse_loss(inputs, outputs)
        # loss = torch.sum((inputs-outputs)**2)
        loss.backward()
        # print(loss)
        # if i > 20:
        preconditioner.step() # g = F^-1 * g
        optimizer.step() # 1. d = A * d; 2. x' = x + alpha * d
        if i%20==0:
            val_loss = evaluate(model, F.mse_loss, test_loader).data
            print(f'iteration:{i}, loss: {val_loss}')
    return model

auto_encoder_MNIST()