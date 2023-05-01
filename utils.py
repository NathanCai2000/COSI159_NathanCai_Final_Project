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
        
        
        
import tensorflow as tf

def median_filter(input_tensor, filter_size):
    # Pad the input tensor to handle edges of the image.
    padding_size = filter_size // 2
    padded_tensor = tf.pad(input_tensor, [[0, 0], [padding_size, padding_size], [padding_size, padding_size], [0, 0]], 'REFLECT')

    # Create a sliding window of size filter_size x filter_size.
    kernel = tf.ones([filter_size, filter_size, input_tensor.get_shape().as_list()[-1], 1], dtype=tf.float32)

    # Apply median filtering using the sliding window and the median() function.
    filtered_tensor = tf.nn.depthwise_conv2d(padded_tensor, kernel, strides=[1, 1, 1, 1], padding='VALID')
    filtered_tensor = tf.squeeze(filtered_tensor, axis=-1)
    filtered_tensor = tf.transpose(filtered_tensor, [0, 3, 1, 2])
    filtered_tensor = tf.map_fn(lambda x: tf.contrib.distributions.percentile(x, 50.0), filtered_tensor, dtype=tf.float32)
    filtered_tensor = tf.transpose(filtered_tensor, [0, 2, 3, 1])

    return filtered_tensor

        
        
        
        
        
        
        
        
        
        