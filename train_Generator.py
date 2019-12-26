import torch
import torch.nn as nn
# import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
import numpy as np
import data
import net


is_debug = True
log = "./model/log_maxiter20_g_grad0.txt"

def pseudo_update_f(f,f_copy,dataset_name):
    if dataset_name == 'MNIST':
        tmp = 0
        for i in range(len(list(f.parameters()))):
            list(f_copy.parameters())[i].data = list(f.parameters())[i].data - f.lr*list(f.parameters())[i].grad





def train_generator(dataset,g,T,maxiter,lr_f,lr_g,if_use_gpu):

    g.train()

    # g_copy = g
    torch.save(g, "g_tmp.pkl")
    g_copy = torch.load('g_tmp.pkl')
    g_copy.train()
    g_copy.set_lr_opt(lr_g)



    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("CUDA is not available, fall back to CPU.")

    g_copy = g_copy.to(device)
    g_copy.loss = g_copy.loss.to(device)



    g = g.to(device)
    g.loss = g.loss.to(device)

    logfile = open(log,'w')


    for t in range(T):
        f = net.CNN_Origin_MNIST()
        f.train()
        f.set_lr_opt(lr_f)

        # f_copy = f
        torch.save(f, "f_tmp.pkl")
        f_copy = torch.load('f_tmp.pkl')
        f_copy.train()
        f_copy.set_lr_opt(lr_f)

        f_copy = f_copy.to(device)
        f_copy.loss = f_copy.loss.to(device)

        f = f.to(device)
        f.loss = f.loss.to(device)

        for i in range(maxiter): # epoch
            # loss_mean = 0
            train_mean_loss = 0
            train_acc = 0
            for x,y in dataset.trainloader:

                x = Variable(x, requires_grad=False)
                y = Variable(y, requires_grad=False)

                if if_use_gpu:
                    x = x.cuda()
                    y = y.cuda()
                g_copy.opt.zero_grad()
                noise = g_copy(x)

                f.opt.zero_grad()
                f_copy.opt.zero_grad()
                f_copy_loss = f.loss(f(x + noise), y)
                f_copy_loss.backward()

                # Update g' using current fθ
                # seta_tmp = f.parameters() - f.lr * f.parameters().grad
                pseudo_update_f(f, f_copy, 'MNIST')

                g_loss = - g_copy.loss(f_copy(x),y)
                g_loss.backward()
                g_copy.opt.step()

                # Update fθ by SGD
                f.opt.zero_grad()
                x_adversarial = x + g(x)

                out = f(x_adversarial)

                f_loss = f.loss(out,y)

                train_mean_loss += f_loss.cpu()
                _, pred = out.max(1)
                num_correct = (pred.cpu() == y.cpu()).sum().item()
                acc = num_correct / x.shape[0]
                train_acc += acc

                f_loss.backward()
                f.opt.step()



            # test f
            eval_mean_loss = 0
            eval_acc = 0
            for test_x,test_y in dataset.testloader:
                f.eval()
                test_x = Variable(test_x, requires_grad=False)
                test_y = Variable(test_y, requires_grad=False)
                if if_use_gpu:
                    test_x = test_x.cuda()
                    test_y = test_y.cuda()
                out = f(test_x)
                eval_loss = f.loss(out,test_y)
                eval_mean_loss += eval_loss
                _, pred = out.max(1)
                num_correct = (pred.cpu() == test_y.cpu()).sum().item()
                acc = num_correct/test_x.shape[0]
                eval_acc += acc
                # accuracy = torch.max(out.cpu(), 1)[1].numpy() == test_y.cpu().numpy()
            print('t: {}, i: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Eval Loss: {:.3f}, Eval Acc: {:.3f}'
                  .format(t,i, train_mean_loss / len(dataset.trainloader), train_acc / len(dataset.trainloader),
                          eval_mean_loss / len(dataset.testloader), eval_acc / len(dataset.testloader)))
            if t%5 == 0:
                logfile.write('t: {}, i: {}, Train Loss: {:.3f}, Train Acc: {:.3f}, Eval Loss: {:.3f}, Eval Acc: {:.3f}\n'
                  .format(t,i, train_mean_loss / len(dataset.trainloader), train_acc / len(dataset.trainloader),
                          eval_mean_loss / len(dataset.testloader), eval_acc / len(dataset.testloader)))

        if t == 75:
            lr_g /= 100

        # g = g_copy
        torch.save(g_copy, "g_copy_tmp.pkl")
        g = torch.load('g_copy_tmp.pkl')
        g.train()
        g.set_lr_opt(lr_g)

        if t%10 == 0:
            torch.save(g,'./model/g_t_{}.pkl'.format(t))



    torch.save(g, "./model/g.pkl")
    logfile.close()



