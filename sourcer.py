# -*- coding: utf-8 -*-
"""
@author: Nathan Cai
"""

import argparse

import torch
import torchvision
import os

from model import Net
from train import Trainer
from utils import RandomNoiseDataset
import Gen_Mod as GM
import GAN_Mod as GA
import FE_Mod as FE
import mnist_m as mm


def parse_args():
    parser = argparse.ArgumentParser(description='Domain Impression')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=100, help="batch size")
    args = parser.parse_args()

    return args

def main():
    
    """Part 1: Source Classifier data Generator--------------------------------"""
    args = parse_args()
    # model
    model_source = Net()
    
    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([28, 28]),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor(),
        #torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    #MNIST
    # target_set = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    # target_train = torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform)
    # name = "mnist"
    
    #MNIST-M
    # target_set = mm.MNISTM(root='./data/', train=True, download=True, transform=transform)
    # target_train = mm.MNISTM(root='./data/', train=False, download=True, transform=transform)
    # name = "mnist_m"
    
    
    #SVHN
    target_set = torchvision.datasets.SVHN(root='./data/', split='train', download=True, transform=transform)
    target_train = torchvision.datasets.SVHN(root='./data/', split='test', download=True, transform=transform)
    name = "SVHN"
    
    #USPS
    # target_set = torchvision.datasets.USPS(root='./data/', train=True, download=True, transform=transform)
    # target_train = torchvision.datasets.USPS(root='./data/', train=False, download=True, transform=transform)
    # name = "USPS"
    
    #Dataloader    
    train_loader = torch.utils.data.DataLoader(
        target_set,
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        target_train,
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer 
    trainer = Trainer(model=model_source) 
    
    # model training for source classifier
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/", name=name)
    
    trainer.eval(test_loader)
    
    return

if __name__ == "__main__":
    main()
    
    