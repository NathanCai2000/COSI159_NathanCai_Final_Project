# -*- coding: utf-8 -*-
"""
@author: Nathan Cai
"""

import math
from utils import AverageMeter
import time
import os
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

import matplotlib.pyplot as plt



class Generator(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(110, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.Linear(512, 784),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(1024, 784 *3),
            nn.Tanh()
        )
    
    def forward(self, z, labels):
        z = z.view(z.size(0), 100)
        c = self.label_emb(labels)
        x = torch.cat([z, c], 1)
        out = self.model(x)
        return out.view(z.size(0), 3, 28, 28)

class Source_gen:
    
    def __init__(self, source_model: nn.modules):
        self.gen = Generator()
        self.source = source_model

    def train(self,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        bs: int,
        save_dir: str,
        name: str,
    ) -> None:
            "Trains the Generator for the Source Data GAN framework"
            optimizer = optim.SGD(params=self.gen.parameters(), lr=lr)
            loss_track = AverageMeter()
            criterion = nn.CrossEntropyLoss()

            
            self.gen.train()
            print("Start training...")
            for i in range(epochs):
                tik = time.time()
                loss_track.reset()
                #for a in range(sample_n):
                for data, label in train_loader:
                    
                    optimizer.zero_grad()
                    z = Variable(data)
                    label = Variable(label)
                    generated_image = self.gen(z, label)
                    output = self.source(generated_image)

                    #loss_CEL = criterion(output, label)
                    loss_CEL = criterion(output, label)

                    loss_LSE = torch.logsumexp(output, 1)
                    loss = loss_LSE + loss_CEL
                    #print(loss)
                    loss.backward()
                    optimizer.step()
                    
                    loss_track.update(loss.item(), n=1)
                    
                elapse = time.time() - tik
                stats = "Epoch: [%d/%d]; Time: %.2f; Loss: %.5f" % (i + 1, epochs, elapse, loss_track.avg)
                print(stats)
                self.eval(9, stats)
                
            print("Training completed, saving model to %s" % save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.gen.state_dict(), os.path.join(save_dir, name + ".pth"))
        
            return
    
    def eval(self, sample_n, title=''):
        """Model testing evaluation: Generates n number of fake samples and displays it."""
        store = int(math.sqrt(sample_n))
        [r, c] = store, store
        
        plt.figure()
        for a in range(sample_n):
            z = Variable(torch.randn(1, 100))
            label = Variable(torch.LongTensor(np.random.randint(0, 10, 1)))
            
            output = self.gen(z, label)
                        
            output = output[0].permute(1, 2, 0).detach().numpy()
            
            output = (output * 255).astype(np.uint8)
            
            plt.subplot(r,c, a+1)
            plt.tight_layout()
            plt.imshow(output)
            plt.title("Prediction: {}".format(label.item()))
        plt.suptitle(title)
        plt.show()
            
    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        print("loading Source Generator Model")
        # loads a model from a .pth file and sets it as the current model
        self.gen.load_state_dict(torch.load(path))
        # Runs an evaluation after loading to ensure operation
        self.gen.eval()
        return




















