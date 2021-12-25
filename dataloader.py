import torch
import torch.nn as nn
from torch.autograd import Variable
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim

# use cuda or not
use_cuda = torch.cuda.is_available()

def get_loader_MNIST(batch_size=1):
    trans = transforms.Compose([transforms.ToTensor()])

    root = './datasets'
    train_set = dset.MNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.MNIST(root=root, train=False, transform=trans)

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False)
    return train_loader, test_loader

if __name__ == '__main__':
    get_loader_MNIST()