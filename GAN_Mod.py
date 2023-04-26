# -*- coding: utf-8 -*-
"""
@author: Nathan Cai
"""
import math
import time
import os
import torch
from torch import optim
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable

import numpy as np

import matplotlib.pyplot as plt

#from Gen_Mod import Generator

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.label_emb = nn.Embedding(10, 10)
        
        self.model = nn.Sequential(
            nn.Linear(794, 1024),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x, labels):
        x = x.view(x.size(0), 784)
        c = self.label_emb(labels)
        x = torch.cat([x, c], 1)
        out = self.model(x)
        return out.squeeze()


class GAN:
    
    def __init__(self, source_gen: nn.modules):
        self.dis = Discriminator()
        self.gen = source_gen
        
    def generator_train_step(self, batch_size, discriminator, generator, g_optimizer, criterion):
        g_optimizer.zero_grad()
        z = Variable(torch.randn(batch_size, 100))
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size)))
        fake_images = generator(z, fake_labels)
        validity = discriminator(fake_images, fake_labels)
        g_loss = criterion(validity, Variable(torch.ones(batch_size)))
        g_loss.backward()
        g_optimizer.step()
        return g_loss.item()
    
    def discrimin_train_step(self, batch_size, discriminator, generator, d_optimizer, criterion, real_images, labels):
        d_optimizer.zero_grad()
    
        # train with real images
        real_validity = discriminator(real_images, labels)
        real_loss = criterion(real_validity, Variable(torch.ones(batch_size)))
        
        # train with fake images
        z = Variable(torch.randn(batch_size, 100))
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, batch_size)))
        fake_images = generator(z, fake_labels)
        fake_validity = discriminator(fake_images, fake_labels)
        fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)))
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        return d_loss.item()
        
    def train(self,
        train_loader: DataLoader,
        epochs: int,
        lr: float,
        bs: int,
        save_dir: str,
        name: str,
    ) -> None:
            "Trains the Generator for the Source Data GAN framework"
            
            criterion = nn.BCELoss()
            d_optimizer = optim.SGD(params=self.dis.parameters(), lr=0.01)
            g_optimizer = optim.SGD(params=self.gen.parameters(), lr=lr)

            
            print("Start training...")
            
            for epoch in range(epochs):
                tik = time.time()
                print('Starting epoch {}...'.format(epoch+1))
                for i, (images, labels) in enumerate(train_loader):
                    real_images = Variable(images)
                    labels = Variable(labels)
                    self.gen.train()
                    batch_size = real_images.size(0)
                    d_loss = self.discrimin_train_step(batch_size, self.dis, self.gen, d_optimizer, criterion, real_images, labels)
                    
                    g_loss = self.generator_train_step(batch_size, self.dis, self.gen, g_optimizer, criterion)
                
                elapse = time.time() - tik
                print("Epoch: [%d/%d]; Time: %.2f; g_Loss: %.5f; d_loss: %.5f" % (epoch+1, epochs, elapse, g_loss, d_loss))
                
                self.eval(9)
                
            print("Training completed, saving model to %s" % save_dir)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            torch.save(self.gen.state_dict(), os.path.join(save_dir, name + ".pth"))
            
    
        
    def eval(self, sample_n):
        """Model testing evaluation: Generates n number of fake samples and displays it."""
        store = int(math.sqrt(sample_n))
        [r, c] = store, store
        
        fig = plt.figure()
        for a in range(sample_n):
            z = Variable(torch.randn(1, 100))
            label = Variable(torch.LongTensor(np.random.randint(0, 10, 1)))
            
            output = self.gen(z, label)
            output = output.permute(1, 2, 0).detach().numpy()
            
            plt.subplot(r,c, a+1)
            plt.tight_layout()
            plt.imshow(output, cmap='gray')
            plt.title("Prediction: {}".format(label.item()))
            
        fig
        
    def load_model(self, path: str) -> None:
        """ load model from a .pth file """
        print("loading Source Generator Model")
        # loads a model from a .pth file and sets it as the current model
        self.gen.load_state_dict(torch.load(path))
        # Runs an evaluation after loading to ensure operation
        self.gen.eval()
        return