import torch
from torch._C import _llvm_enabled
import torch.optim as optim
import torch.nn as nn
import itertools, functools
from torch.nn import functional as F
from torch.optim import optimizer
from model import MLPNet, CNN, AEMNIST
from dataloader import get_loader_MNIST
from ekfac import EKFAC
from kfac import KFAC
from temp import test

# Hyper-parameters
lr = 1e-1 # learning rate
batch_size=200
use_precondition = False
epochs = 200

train_loader, test_loader = get_loader_MNIST(batch_size=batch_size)

# Training
def classification_MNIST():
    """toy experiment on MNIST classification"""
    model = CNN()
    preconditioner = KFAC(model, 0.1, update_freq=1)
    optimizer = optim.SGD(model.parameters(), lr=lr)
    for i, (inputs, targets) in enumerate(train_loader):
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = F.cross_entropy(outputs, targets)
        loss.backward()
        print(loss)
        preconditioner.step()
        optimizer.step()
    return model

def evaluate(model, criterion, dataloader):
    """evaluate auto-encoder"""
    model.eval() # change BN
    loss = 0.
    length = 0
    for test_data, label in dataloader:
        out = model(test_data).view(test_data.size())
        loss += criterion(out, test_data)
        length += 1
    model.train()
    return loss / length

def auto_encoder_MNIST(model, optimizer, train_loader, test_loader, exp_name, epochs=30):
    """experiment on autoencoder"""
    # model = AEMNIST()
    # optimizer = optim.SGD(model.parameters(), lr=lr)
    # preconditioner = KFAC(model, 0.1, update_freq=1)
    for epoch in range(epochs):
        train_loss = evaluate(model, F.mse_loss, train_loader).data
        val_loss = evaluate(model, F.mse_loss, test_loader).data
        log = f'epoch:{epoch}, train loss: {train_loss:8.6f}, val loss: {val_loss:8.6f}'
        with open(f'./experiments/{exp_name}.txt', 'a') as f:
            f.write(log + '\n')
        print(log)
        for i, (inputs, _) in enumerate(train_loader):
            optimizer.zero_grad()
            outputs = model(inputs).view(inputs.size())
            loss = F.mse_loss(inputs, outputs)
            loss.backward()
            # if i % 20 == 0:
            #     print(f'epoch:{epoch}, iter:{i}, batch loss:{loss:8.6f}')
            # preconditioner.step() # g = F^-1 * g
            optimizer.step()
    return model

def exp1():
    """Experiment on three hyperparameters:
        * step size 
        * batch size 
        * optimizer
    """
    epochs = 30
    bsizes = [200, 500]
    lrs = [1e-1, 1e-2, 1e-3, 1e-4]
    optimizers = {
        'SGD': optim.SGD,
        'Momentum': functools.partial(optim.SGD, momentum=0.9),
        'Adam': optim.Adam,
        'Adadelta': optim.Adadelta
    }
    bz_lr_opt = itertools.product(bsizes, lrs, optimizers.items())
    for bz, lr, opt_item in bz_lr_opt:
        opt_name, opt = opt_item
        exp_name = f'bz_{bz}_lr_{lr}_opt_{opt_name}'
        model = AEMNIST()
        dataloaders = get_loader_MNIST(batch_size=bz)
        optimizer = opt(model.parameters(), lr=lr)
        auto_encoder_MNIST(model, optimizer, *dataloaders, exp_name, epochs)

exp1()