import ipdb
import os
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
from utils import evaluate

use_cuda = torch.cuda.is_available()

def evaluate(model, criterion, dataloader):
    """evaluate auto-encoder"""
    model.eval() # change BN
    loss = 0.
    length = 0
    for test_data, label in dataloader:
        if use_cuda:
            test_data, label = test_data.cuda(), label.cuda()
        out = model(test_data).view(test_data.size())
        loss += criterion(out, test_data)
        length += 1
    model.train()
    return loss / length


def auto_encoder_MNIST(model, optimizer, train_loader, test_loader, exp_name, epochs=30, preconditioner=None, save_loss=True):
    """experiment on autoencoder"""
    losses = []
    for epoch in range(epochs):
        # evaluation
        train_loss = evaluate(model, F.mse_loss, train_loader).data
        val_loss = evaluate(model, F.mse_loss, test_loader).data
        log = f'epoch:{epoch}, train loss: {train_loss:8.6f}, val loss: {val_loss:8.6f}'
        with open(f'./experiments/{exp_name}.txt', 'a') as f:
            f.write(log + '\n')
        print(log)

        # save loss
        if save_loss:
            losses.append((train_loss, val_loss))
            if (epoch+1) % 10 == 0:
                torch.save(losses, f'./exp_losses/ep_{epoch}.pt' + exp_name)

        # train for one epoch
        for i, (inputs, _) in enumerate(train_loader):
            if use_cuda:
                inputs = inputs.cuda()
            optimizer.zero_grad()
            outputs = model(inputs).view(inputs.size())
            loss = F.mse_loss(inputs, outputs)
            loss.backward()
            
            if i % 20 == 0:
                print(f'epoch:{epoch}, iter:{i}, batch loss:{loss:8.6f}')
            if preconditioner is not None:
                preconditioner.step()
            optimizer.step()
    
    return model

def exp1():
    """Experiment on three hyperparameters on MNIST auto-encoder:
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
        if use_cuda: model = model.cuda()
        dataloaders = get_loader_MNIST(batch_size=bz)
        optimizer = opt(model.parameters(), lr=lr)
        auto_encoder_MNIST(model, optimizer, *dataloaders, exp_name, epochs)

def exp2():
    """Experiments on finding the optimal lr and eps for KFAC/EKFAC (on MNIST):
    """
    epochs = 30
    batch_size = 500
    eps = [1e-4, 1e-3, 0]
    lrs = [1e-3, 1e-2, 1e-1, 1]
    update_freq = [1, 20, 50]
    preconditioners = {
        # 'KFAC': KFAC, 
        'EKFAC': EKFAC,
        # 'EKFAC-ra': functools.partial(EKFAC, ra=True, alpha=0.9)
    }
    ep_lr_pre = itertools.product(eps, update_freq, lrs, preconditioners.items())
    for ep, up_freq, lr, pre_item in ep_lr_pre:
        pre_name, pre = pre_item
        # exp_name = 'test'
        exp_name = f'exp2_{pre_name}_ep_{ep}_lr_{lr}_freq_{up_freq}'
        if os.path.isfile(f'./experiments/{exp_name}'): os.remove(f'./experiments/{exp_name}')
        model = AEMNIST()
        preconditioner = pre(model, eps=ep, update_freq=up_freq)
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
        if use_cuda: model = model.cuda()
        dataloaders = get_loader_MNIST(batch_size=batch_size)
        auto_encoder_MNIST(model, optimizer, *dataloaders, exp_name, preconditioner=preconditioner, epochs=epochs)

def exp3():
    """Experiment on comparing the following optimization methods (on MNIST):
    * SGD + Momentum
    * Adam
    * KFAC: freq 50
    * EKFAC: freq 50
    * EKFAC-ra: freq 50
    The criterion is the training epoch error and wall-clock time
    NOTE: remember to show the validation loss.
    NOTE: remember to save the tensor for visualization
    """
    epochs = 100
    batch_size = 500
    exps = [
        {'name': 'SGD', 'optimizer': functools.partial(optim.SGD, momentum=0.9), 'lr': 1e-3, 'precond': None},
        {'name': 'Adam', 'optimizer': optim.Adam, 'lr': 1e-3, 'precond': None},
        {'name': 'KFAC50', 'optimizer': functools.partial(optim.SGD, momentum=0.9), 'lr': 1e-3, 'precond': functools.partial(KFAC, update_fre=50), 'eps': 1e-3},
        {'name': 'EKFAC50', 'optimizer': functools.partial(optim.SGD, momentum=0.9), 'lr': 1e-3, 'precond': functools.partial(EKFAC, update_fre=50), 'eps': 1e-3},
        {'name': 'EKFAC-ra50', 'optimizer': functools.partial(optim.SGD, momentum=0.9), 'lr': 1e-3, 'precond': functools.partial(EKFAC, update_fre=50, ra=True, alpha=0.9), 'eps': 1e-3},
        # {'name': 'KFAC1', 'optimizer': functools.partial(optim.SGD, momentum=0.9), 'lr': 1e-3, 'precond': functools.partial(KFAC, update_fre=1, eps=1e-3)},
        # {'name': 'EKFAC1', 'optimizer': functools.partial(optim.SGD, momentum=0.9), 'lr': 1e-3, 'precond': functools.partial(EKFAC, update_fre=1, eps=1e-3)},
        # {'name': 'EKFAC-ra1', 'optimizer': functools.partial(optim.SGD, momentum=0.9), 'lr': 1e-3, 'precond': functools.partial(EKFAC, update_fre=1, eps=1e-3, ra=True, alpha=0.9)},
    ]
    for runs in exps:
        pre_name, opt, lr, pre, eps = runs['name'], runs['optimizer'], runs['lr'], runs['precond'], runs['eps']
        exp_name = f'exp3_{pre_name}_lr_{lr}_ep_{eps}'
        if os.path.isfile(f'./experiments/{exp_name}'): os.remove(f'./experiments/{exp_name}')
        
        model = AEMNIST()
        if use_cuda: model = model.cuda()
        preconditioner = None
        if pre is not None:
            preconditioner = pre(model, eps=eps)
        dataloaders = get_loader_MNIST(batch_size=batch_size)
        optimizer = opt(model.parameters(), lr=lr)
        
        auto_encoder_MNIST(model, optimizer, *dataloaders, exp_name, preconditioner=preconditioner, epochs=epochs)

def exp4():
    """Experiment on the impact of frequency (on MNIST). 
    The following experiments are conducted:
    * KFAC: freq 1/50/100
    * EKFAC: freq 50/100
    The criterion is the training epoch error and wall-clock time
    NOTE: if time permitted, show the distance to spectrum of G.
    """
    pass

def exp5():
    """Experiment on the classification error of CIFAR using VGG-16.
    The following experiments are conducted:
    * SGD Momentum (with BN)
    * KFAC: freq 500
    * EKFAC: freq 500
    * EKFAC: freq 500
    The criterion is the training epoch error and wall-clock time
    NOTE: no BN used for KFAC/EKFAC
    """
    pass

exp2()