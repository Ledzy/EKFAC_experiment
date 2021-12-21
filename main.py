import torch
import torch.optim as optim
import torch.nn as nn
from torch.nn import functional as F
from model import MLPNet, CNN
from dataloader import get_loader_MNIST
from ekfac import EKFAC
from kfac import KFAC

# Hyper-parameters
lr = 1 # learning rate
batch_size=64

model = CNN()
preconditioner = KFAC(model, 0.1, update_freq=1)
optimizer = optim.SGD(model.parameters(), lr=lr)
train_loader, test_loader = get_loader_MNIST(batch_size=batch_size)

# Training
for i, (inputs, targets) in enumerate(train_loader):
    optimizer.zero_grad() # set tensor's grad attribute to 0
    outputs = model(inputs)
    loss = F.cross_entropy(outputs, targets)
    loss.backward()
    print(loss)
    preconditioner.step() # g = F^-1 * g
    optimizer.step() # 1. d = A * d; 2. x' = x + alpha * d