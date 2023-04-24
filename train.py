import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F

from utils import AverageMeter


import matplotlib.pyplot as plt


class Trainer:
    """ Trainer for MNIST classification """

    def __init__(self, model: nn.Module):
        self._model = model

    def train(
            self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            save_dir: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer = optim.SGD(params=self._model.parameters(), lr=lr)
        loss_track = AverageMeter()
        self._model.train()

        print("Start training...")
        for i in range(epochs):
            tik = time.time()
            loss_track.reset()
            for data, target in train_loader:
                optimizer.zero_grad()
                output = self._model(data)

                loss = F.nll_loss(output, target)
                loss.backward()
                optimizer.step()

                loss_track.update(loss.item(), n=data.size(0))

            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg))

        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self._model.state_dict(), os.path.join(save_dir, "mnist.pth"))

        return

    def eval(self, test_loader: DataLoader) -> float:
        """ Model evaluation, return the model accuracy over test set """
        
        print("Evaluating:...")
        
        self._model.eval()
        
        test_loss = 0
        correct = 0

        for data, target in test_loader:
          output = self._model(data)
          test_loss += F.nll_loss(output, target, size_average=False).item()
          pred = output.data.max(1, keepdim=True)[1]
          correct += pred.eq(target.data.view_as(pred)).sum()
        test_loss /= len(test_loader.dataset)
        
        print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
          test_loss, correct, len(test_loader.dataset),100. * correct / len(test_loader.dataset)))
        
        return

    def infer(self, sample: Tensor) -> int:
        """ Model inference: input an image, return its class index """
        # inputs the sample to the model
        output = self._model(sample)
        
        # utility code to help display the image and the predictions in sypder plots
        fig = plt.figure()
        for i in range(6):
          plt.subplot(2,3,i+1)
          plt.tight_layout()
          plt.imshow(sample[i][0], cmap='gray', interpolation='none')
          plt.title("Prediction: {}".format(output.data.max(1, keepdim=True)[1][i].item()))
        fig
        
        # prints out the associated classes for each sample
        print(output.data.max(1, keepdim=True)[1][:])
        return 

    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        # loads a model from a .pth file and sets it as the current model
        self._model.load_state_dict(torch.load(path))
        # Runs an evaluation after loading to ensure operation
        self._model.eval()
        return

