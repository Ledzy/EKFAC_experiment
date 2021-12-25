import torch
import torch.nn as nn
import torch.nn.functional as F

class MLPNet(nn.Module):
    def __init__(self):
        super(MLPNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 500)
        self.fc2 = nn.Linear(500, 256)
        self.fc3 = nn.Linear(256, 10)
    def forward(self, x):
        # remark on view: dim of x: (12,3,3) -> dim of x.view(-1, 4) is (12*3*3/4, 4)
        x = x.view(-1, 28*28) # change the size of x: (batch_size, 28, 28) -> (batch_size, 28*28)
        x = F.relu(self.fc1(x)) # x <- 
        x = F.relu(self.fc2(x))
        x = F.softmax(self.fc3(x))
        return x

class AEMNIST(nn.Module):
    def __init__(self):
        super(AEMNIST, self).__init__()
        units = [28*28, 1000, 500, 250, 30, \
                    250, 500, 1000, 28*28]
        self.fc_layers = nn.ModuleList([nn.Linear(units[i], units[i+1]) for i in range(len(units)-1)])
        # for i in range(len(units)-1):
        #     self.fc_layers.append(nn.Linear(units[i], units[i+1]))
    
    def forward(self, x):
        x = x.view(-1, 28*28)
        for fc in self.fc_layers:
            x = F.sigmoid(fc(x))
        return x

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)
        return output