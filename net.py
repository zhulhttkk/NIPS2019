import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

class CNN_Origin_CIFAR10(nn.Module):
    def __init__(self):
        super(CNN_Origin_CIFAR10, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)
        self.conv5 = nn.Conv2d(128, 128, kernel_size=3)
        self.fc = nn.Linear(500,10)

    def forward(self,x):
        x = F.max_pool2d(self.conv1(x),2)
        x = F.max_pool2d(self.conv2(x), 2)
        x = F.max_pool2d(self.conv3(x), 2)
        x = F.max_pool2d(self.conv4(x), 2)
        x = F.max_pool2d(self.conv5(x), 2)
        x = x.view(-1,500)
        x = self.fc(x)
        return F.relu(x)


# f
class CNN_Origin_MNIST(nn.Module):
    def __init__(self):
        super(CNN_Origin_MNIST, self).__init__()
        self.conv1 = nn.Conv2d(1, 20, kernel_size=3)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=3)
        self.fc1 = nn.Linear(50*24*24,500)
        self.fc2 = nn.Linear(500, 10)

    def forward(self,x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1,50*24*24)
        x = self.fc1(x)
        x = self.fc2(x)
        return F.relu(x)

    def set_lr_opt(self,lr):
        self.lr = lr
        self.loss = nn.CrossEntropyLoss()
        self.opt = optim.SGD(self.parameters(), lr=self.lr)

# g
class Generator_MINST(nn.Module):
    def __init__(self,lr,constraint):
        super(Generator_MINST, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=4),
            nn.Conv2d(16, 32, kernel_size=4),
            nn.Conv2d(32, 64, kernel_size=4),
            nn.Conv2d(64, 128, kernel_size=4)
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4),
            nn.ConvTranspose2d(64, 32, kernel_size=4),
            nn.ConvTranspose2d(32, 16, kernel_size=4),
            nn.ConvTranspose2d(16, 1, kernel_size=4)
        )
        self.loss = nn.CrossEntropyLoss()
        # self.lr = lr
        # self.opt = optim.Adam(self.parameters(), lr=self.lr)
        self.constraint = constraint
    def forward(self,x):
        x = self.encoder(x)
        x = self.decoder(x)
        x = torch.tanh(x)
        x = self.constraint*x
        return x

    def set_lr_opt(self,lr):
        self.lr = lr
        self.opt = optim.Adam(self.parameters(), lr=self.lr)






