# -*- coding: utf-8 -*-
"""

@author: Nathan Cai
"""

import os
import time

import torch
from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np


from utils import AverageMeter


import matplotlib.pyplot as plt

class GradientReversalLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output.neg()
    
class DA_dis(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.model = nn.Sequential(
            nn.Linear(64, 1024),
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
    
    def forward(self, x):
        x = x.view(x.size(0), 64)
        out = self.model(x)
        return out.squeeze()

class FeatureExtractor(nn.Module):
    def __init__(self):
        super(FeatureExtractor, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=5, padding=2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=5, padding=2)
        self.fc1 = nn.Linear(32*7*7, 512)
        self.fc2 = nn.Linear(512, 64)
        self.grl = GradientReversalLayer.apply
        
    def forward(self, x, alpha=1.0):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 32*7*7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = self.grl(x) * alpha
        return x

class FEDA_Trainer():
    def __init__(self, model: nn.Module):
        self.gen = model
        self.FE = FeatureExtractor()
        self.DA = DA_dis()
        
        
    
    def FE_train_step(self, batch_size, DA, FE, g_optimizer, criterion):
        g_optimizer.zero_grad()
        z = Variable(torch.randn(1, 100))
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, 1)))
        fake_image = self.gen(z, fake_labels)
        fake = FE(fake_image)
        validity = DA(fake)
        g_loss = criterion(validity, Variable(torch.ones(batch_size)))
        g_loss.backward()
        g_optimizer.step()
        return g_loss.item()
    
    def DA_train_step(self, batch_size, DA, FE, d_optimizer, criterion, real_image):
        d_optimizer.zero_grad()
    
        # train with real images
        real = FE(real_image)
        print(real.size())
        real_validity = DA(real)
        real_loss = criterion(real_validity, Variable(torch.ones(batch_size)))
        
        # train with fake images
        z = Variable(torch.randn(1, 100))
        fake_labels = Variable(torch.LongTensor(np.random.randint(0, 10, 1)))
        fake_image = self.gen(z, fake_labels)
        fake = FE(fake_image)
        fake_validity = DA(fake)
        fake_loss = criterion(fake_validity, Variable(torch.zeros(batch_size)))
        
        d_loss = real_loss + fake_loss
        d_loss.backward()
        d_optimizer.step()
        return d_loss.item()
    
    def train(
            self,
            train_loader: DataLoader,
            epochs: int,
            lr: float,
            bs: int,
            save_dir: str,
            name: str,
    ) -> None:
        """ Model training, TODO: consider adding model evaluation into the training loop """

        optimizer_FE = optim.SGD(params=self.FE.parameters(), lr=lr)
        optimizer_DA = optim.SGD(params=self.DA.parameters(), lr=lr)
        
        criterion = nn.CrossEntropyLoss()
        
        loss_track_FE = AverageMeter()
        loss_track_DA = AverageMeter()

        print("Start training...")
        
        for epoch in range(epochs):
            tik = time.time()
            print('Starting epoch {}...'.format(epoch+1))
            for i, (images, labels) in enumerate(train_loader):
                real_image = Variable(images)
                self.gen.train()
                batch_size = real_image.size(0)
                FE_loss = self.FE_train_step(batch_size, self.DA, self.FE, optimizer_FE, criterion)
                
                DA_loss = self.DA_train_step(batch_size, self.DA, self.FE, optimizer_DA, criterion, real_image)
            
            elapse = time.time() - tik
            print("Epoch: [%d/%d]; Time: %.2f; FE_Loss: %.5f; DA_loss: %.5f" % (epoch+1, epochs, elapse, FE_loss, DA_loss))
            
        print("Training completed, saving model to %s" % save_dir)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        torch.save(self.gen.state_dict(), os.path.join(save_dir, name + ".pth"))