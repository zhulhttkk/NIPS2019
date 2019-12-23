import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import data
import net

x = Variable(torch.ones(2, 2), requires_grad = True)

y = Variable(torch.zeros(2, 2), requires_grad = True)

y = x # y = 1


x=Variable(torch.zeros(2, 2)) # x=0

print(x)
print(y)



