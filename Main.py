import argparse

import torch
import torchvision
import os

from model import Net, Net2
from train import Trainer
from utils import RandomNoiseDataset, GenSet
import Gen_Mod as GM
import GAN_Mod as GA
import FE_Mod as FE
import Adapt_Mod as AP
import mnist_m as mm
from scipy import ndimage


def parse_args():
    parser = argparse.ArgumentParser(description='Domain Impression')
    parser.add_argument('--epochs', type=int, default=5, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=100, help="batch size")
    args = parser.parse_args()

    return args


def main():
    
    """Part 1: Source Classifier data Generator--------------------------------"""
    args = parse_args()
    # model
    model_source = Net()

    # trainer 
    trainer = Trainer(model=model_source) 
    
    # loads the source model
    print("Loading Source Model")
    trainer.load_model(os.path.join("./save/", "mnist.pth"))
    model_source = trainer._model
    
    #Creates a new Generative Model that is trained using the Pretrained Model that was trained on the Source Data
    sample_n = 10000
    dataset = RandomNoiseDataset(sample_n)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)
    
    maker = GM.Source_gen(source_model=model_source)
    #maker.train(train_loader=dataloader, epochs=args.epochs, lr=args.lr / 10, bs=args.bs, save_dir="./save/", name="GAN_s_mnist")

    maker.load_model(os.path.join("./save/", "GAN_s_mnist.pth"))
    maker.eval(9, title='Generator Model: Source=MNIST')

    
    """Part 2: Target Domain GAN-----------------------------------------------"""
    
    # Targe dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([28, 28]),
        torchvision.transforms.Grayscale(3),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    #MNIST
    #target_set = torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform)
    
    #MNIST-M
    #target_set = mm.MNISTM(root='./data/', train=True, download=True, transform=transform)
    
    #SVHN
    #target_set = torchvision.datasets.SVHN(root='./data/', split='train', download=True, transform=transform)
    
    #USPS
    target_set = torchvision.datasets.USPS(root='./data/', train=True, download=True, transform=transform)
    
    #MNIST-M
    train_loader = torch.utils.data.DataLoader(
        target_set,
        batch_size=args.bs,
        shuffle=True,
    )
    
    GAN = GA.GAN(source_gen=maker.gen)
    #GAN.train(train_loader, epochs=args.epochs, lr=args.lr*10, bs=args.bs, save_dir="./save/", name="GAN_t_mnist2usps")
    GAN.load_model(os.path.join("./save/", "GAN_t_mnist2usps.pth"))
    GAN.eval(16, title='Generator Model: Trained with Targets')
    
    """Part 3: Feature Extractor in Target Domain------------------------------"""
    
    #Create final feature extractor
    Feature = FE.FEDA_Trainer(model=GAN.gen)
    #Feature.train(train_loader, epochs=args.epochs, lr=args.lr*1, bs=args.bs, save_dir="./save/", name="FE_t_mnist2usps")
    Feature.load_model(os.path.join("./save/", "FE_t_mnist2usps.pth"))
    
    """Part 4: Final Classifier Training---------------------------------------"""
    
    retu = Net2()
    
    gen_n = 6000
    
    gen_set = GenSet(gen_n, maker.gen)
    
    train_loader = torch.utils.data.DataLoader(
        gen_set,
        batch_size=args.bs,
        shuffle=True,
    )
    
    test_loader = torch.utils.data.DataLoader(
        target_set,
        batch_size=args.bs,
        shuffle=False,
    )
    
    retu_trainer = AP.Trainer(model=retu, fe=Feature.FE)
    
    # model training
    #retu_trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/", name="C_t_mnist-usps")

    # model evaluation
    #retu_trainer.eval(test_loader=test_loader)
    

    return
    

if __name__ == "__main__":
    main()
