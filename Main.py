import argparse

import torch
import torchvision
import os

from model import Net
from train import Trainer
import Gen_Mod as GM


def parse_args():
    parser = argparse.ArgumentParser(description='Domain Impression')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=100, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()
    # model
    model_source = Net()

    # trainer
    trainer = Trainer(model=model_source)
    '''
    print("Loading Data for Source Domain: SVHN")
    # source datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize([28, 28]),
        torchvision.transforms.Grayscale(),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_source = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(root='./data/', split='train', download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_source = torch.utils.data.DataLoader(
        torchvision.datasets.SVHN(root='./data/', split='test', download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )
    '''
    # loads the source model
    print("Loading Source Model")
    trainer.load_model(os.path.join("./save/", "SVHN.pth"))
    model_source = trainer._model
    
    GAN_s = GM.Generator().cuda
    maker = GM.Source_gen(source_model=model_source)
    maker.train(epochs=args.epochs, lr=args.lr / 10, bs=args.bs, save_dir="./save/", name="GAN_s_SVHN")
    #maker.load_model(os.path.join("./save/", "GAN_s.pth"))
    #maker.eval(9)
    
    
    

    return


if __name__ == "__main__":
    main()
