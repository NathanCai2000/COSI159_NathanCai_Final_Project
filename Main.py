import argparse

import torch
import torchvision
import os

from model import Net
from train import Trainer
import Gen_Mod as GM
import GAN_Mod as GA
import FE_Mod as FE


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

    # trainer 
    trainer = Trainer(model=model_source) 
    # loads the source model
    print("Loading Source Model")
    trainer.load_model(os.path.join("./save/", "SVHN.pth"))
    model_source = trainer._model
    
    #Creates a new Generative Model that is trained using the Pretrained Model that was trained on the Source Data
    maker = GM.Source_gen(source_model=model_source)
    #maker.train(epochs=args.epochs, lr=args.lr / 10, bs=args.bs, save_dir="./save/", name="GAN_s_SVHN")
    maker.load_model(os.path.join("./save/", "GAN_s_SVHN.pth"))
    maker.eval(9)
    
    """Part 2: Target Domain GAN-----------------------------------------------"""
    
    # Targe dataset
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([28, 28]),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    
    GAN = GA.GAN(source_gen=maker.gen)
    #GAN.train(train_loader, epochs=args.epochs, lr=args.lr*1, bs=args.bs, save_dir="./save/", name="GAN_t_mnist")
    GAN.load_model(os.path.join("./save/", "GAN_t_mnist.pth"))
    
    """Part 3: Feature Extractor in Target Domain------------------------------"""
    
    Feature = FE.FEDA_Trainer(model=GAN.gen)
    Feature.train(train_loader, epochs=args.epochs, lr=args.lr*1, bs=args.bs, save_dir="./save/", name="GAN_t_mnist")

    return
    

if __name__ == "__main__":
    main()
