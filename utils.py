import torch

use_cuda = torch.cuda.is_available()

def check_nan(model):
    for name, param in model.named_parameters():
        if torch.isnan(param).any():
            return False
    return True

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