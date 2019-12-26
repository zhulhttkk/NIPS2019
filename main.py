# Learning to Confuse: Generating Training Time Adversarial Data with Auto-Encoder

# main.py

import data
import net
import train_Generator as tg

T = 100
maxiter = 20
lr_f = 0.01
lr_g = 0.0001
bach_size = 64
if_use_gpu = True
constrain = 0.3


if __name__ == "__main__":
    dataset = data.MNIST()
    # f = net.CNN_Origin_MNIST()  # seta
    g = net.Generator_MINST(lr=lr_g, constraint=constrain)  # yipuxilong
    tg.train_generator(dataset,g,T,maxiter,lr_f,lr_g,if_use_gpu)


    