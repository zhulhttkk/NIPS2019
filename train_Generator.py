import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import data
import net

def pseudo_update_f(f,f_copy,dataset_name):
    if dataset_name == 'MNIST':
        tmp = 0
        for i in range(len(list(f.parameters()))):
            list(f_copy.parameters())[i].data = list(f.parameters())[i].data - f.lr*list(f.parameters())[i].grad





def train_generator(dataset,f,g,T,maxiter,lr_f,lr_g,bach_size,if_use_gpu):
    f.train()
    g.train()

    f.set_lr_opt(lr_f)

    # g_copy = g
    torch.save(g, "g_tmp.pkl")
    g_copy = torch.load('g_tmp.pkl')
    g_copy.train()
    g_copy.set_lr_opt(lr_g)

    # f_copy = f
    torch.save(f, "f_tmp.pkl")
    f_copy = torch.load('f_tmp.pkl')
    f_copy.train()
    f_copy.set_lr_opt(lr_f)

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, fall back to CPU.")

    g_copy = g_copy.to(device)
    g_copy.loss = g_copy.loss.to(device)

    f_copy = f_copy.to(device)
    f_copy.loss = f_copy.loss.to(device)

    g = g.to(device)
    g.loss = g.loss.to(device)

    f = f.to(device)
    f.loss = f.loss.to(device)


    for t in range(T):
        for i in range(maxiter):
            for x,y in dataset.trainloader:

                x = Variable(x, requires_grad=False)
                y = Variable(y, requires_grad=False)

                if if_use_gpu:
                    x = x.cuda()
                    y = y.cuda()

                noise = g_copy(x)

                f.opt.zero_grad()
                f_copy.opt.zero_grad()
                f_copy_loss = f.loss(f(x + noise), y)
                f_copy_loss.backward()

                # Update g' using current fθ
                # seta_tmp = f.parameters() - f.lr * f.parameters().grad
                pseudo_update_f(f, f_copy, 'MNIST')
                # g_copy.opt.zero_grad()
                g_loss = - g_copy.loss(f_copy(x),y)
                g_loss.backward()
                g_copy.opt.step()

                # Update fθ by SGD
                f.opt.zero_grad()
                x_adversarial = x + g(x)
                f_loss = f.loss(f(x_adversarial),y)
                f_loss.backward()
                f.opt.step()

                # for debug
                abc = 1+1





