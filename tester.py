# -*- coding: utf-8 -*-
"""
@author: Nathan Cai
"""
import torchvision
from utils import GenSet
import Gen_Mod as GM
import GAN_Mod as GA
import FE_Mod as FE
from model import Net
import os
import matplotlib.pyplot as plt
import torch
import numpy as np

def main():
    
    model_source = Net()
    
    maker = GM.Source_gen(source_model=model_source)
    maker.load_model(os.path.join("./save/", "GAN_s_mnist.pth"))
    maker.eval(4, title='Generator Model Check')
    
    GAN = GA.GAN(source_gen=maker.gen)
    GAN.load_model(os.path.join("./save/", "GAN_t_mnist_m.pth"))
    
    Feature = FE.FEDA_Trainer(model=GAN.gen)
    Feature.load_model(os.path.join("./save/", "FE_t_mnist.pth"))
    
    test = GenSet(9, maker.gen, Feature.FE)
    
    train_loader = torch.utils.data.DataLoader(
        test,
        batch_size=1,
        shuffle=True,
    )
    
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([28, 28]),
        torchvision.transforms.ToTensor(),
    ])
    
    test2 = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(root='./data/', split='train', download=True, transform=transform),
        batch_size=1,
        shuffle=True,
    )
    
    [r, c] = 3, 3
    
    plt.figure()
    
    title = 'Test'
    i = 0
    
    for data, label in test2:
        print(data.size())
        output = data
        output = output[0].permute(1, 2, 0).detach().numpy()
        
        output = (output * 255).astype(np.uint8)
        
        plt.subplot(r,c, i+1)
        plt.tight_layout()
        plt.imshow(output)
        plt.title("Prediction: {}".format(label.item()))
        i += 1
    plt.suptitle(title)
    plt.show()
    
    return



if __name__ == "__main__":
    main()