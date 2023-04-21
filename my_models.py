#import numpy as np
import torch
import torch.nn as nn


class Net0(nn.Module):
    def __init__(self):
        super(Net0, self).__init__()
        self.fc4 = nn.Linear(784,256)
        self.s4 = nn.ReLU()
        self.fc3 = nn.Linear(256,128)
        self.s3 = nn.ReLU()
        self.fc2 = nn.Linear(128,64)
        self.s2 = nn.ReLU()
        self.fc1 = nn.Linear(64,10)
        self.s1 = nn.Softmax(dim = 1)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc4(x)
        x = self.s4(x)
        x = self.fc3(x)
        x = self.s3(x)
        x = self.fc2(x)
        x = self.s2(x)
        x = self.fc1(x)
        x = self.s1(x)
        return x

class Hook():
    def __init__(self, module, backward=False):
        if backward==False:
            self.hook = module.register_forward_hook(self.hook_fn)
        #else:
        #    self.hook = module.register_backward_hook(self.hook_fn)
    def hook_fn(self, module, input, output):
        #self.input = input
        self.output = output
    def close(self):
        self.hook.remove()


