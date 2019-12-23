# Learning to Confuse: Generating Training Time Adversarial Data with Auto-Encoder

# main.py

import data
import net
import train_Generator as tg

T = 10
maxiter = 10
lr_f = 1
lr_g = 1
bach_size = 64
if_use_gpu = True
constrain = 0.1

if __name__ == "__main__":
    dataset = data.MNIST()
    f = net.CNN_Origin_MNIST()  # seta
    g = net.Generator_MINST(lr=lr_g, constraint=constrain)  # yipuxilong
    tg.train_generator(dataset,f,g,T,maxiter,lr_f,lr_g,bach_size,if_use_gpu)


    