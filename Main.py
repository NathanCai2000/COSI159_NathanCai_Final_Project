import argparse

import torch
import torchvision

from model import Net
from train import Trainer


def parse_args():
    parser = argparse.ArgumentParser(description='mnist classification')
    parser.add_argument('--epochs', type=int, default=10, help="training epochs")
    parser.add_argument('--lr', type=float, default=1e-1, help="learning rate")
    parser.add_argument('--bs', type=int, default=64, help="batch size")
    args = parser.parse_args()

    return args


def main():
    args = parse_args()

    # model
    model = Net()

    # datasets
    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=True, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=True,
    )
    test_loader = torch.utils.data.DataLoader(
        torchvision.datasets.MNIST(root='./data/', train=False, download=True, transform=transform),
        batch_size=args.bs,
        shuffle=False,
    )

    # trainer
    trainer = Trainer(model=model)
    
    # model import and test
    """
    import os
    # loads the model again and then evaluate itself again (Should have same scores)
    trainer.load_model(os.path.join("./save/", "mnist.pth"))
    trainer.eval(test_loader=test_loader)
    """

    # model training
    trainer.train(train_loader=train_loader, epochs=args.epochs, lr=args.lr, save_dir="./save/")

    # model evaluation
    trainer.eval(test_loader=test_loader)

    # util that loads the test data for use as the samples for testing model inference
    examples = enumerate(test_loader)
    tmp, (example_data, example_targets) = next(examples)
    example_data.shape
    
    
    # model inference
    sample = example_data  # complete the sample here
    trainer.infer(sample=sample)
    
    

    return


if __name__ == "__main__":
    main()
