class AverageMeter:
    """ Computes and stores the average and current value """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        if self.count > 0:
            self.avg = self.sum / self.count
            
import torch
from torch.utils.data import Dataset
import numpy as np


class RandomNoiseDataset(Dataset):
    def __init__(self, size):
        super(RandomNoiseDataset, self).__init__()
        self.values = torch.Tensor(np.random.rand(size, 100)).type(torch.LongTensor) #Random 100 element long noise vector
        self.labels = torch.Tensor(np.random.randint(0, 10, size)).type(torch.LongTensor) #Random label from 1-10
        self.size = size

    def __len__(self):
        return self.size

    def __getitem__(self, index):
        x = self.values[index]
        y = self.labels[index]
        return x, y
    
    
from torch.autograd import Variable
    
class GenSet(Dataset):
    def __init__(self, size, model):
        super(GenSet, self).__init__()
        self.gen = model
        self.size = size 
        
    def __len__(self):
        return self.size
    
    def __getitem__(self, index):
        x = torch.randn(1, 100)
        y = torch.LongTensor(np.random.randint(0, 10, 1))
        z = Variable(x)
        label = Variable(y)
        
        output = self.gen(z, label)
        
        return output[0], y
        
        
        
        
        
        
        
        
        
        
        
        
        
        