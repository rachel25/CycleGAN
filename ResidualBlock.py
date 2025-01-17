import torch
import torch.nn as nn
from ConvolutionalBlock import ConvolutionalBlock

class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            ConvolutionalBlock(channels, channels, add_activation = True, kernel_size = 3, padding = 1),
            ConvolutionalBlock(channels, channels, add_activation = False, kernel_size = 3, padding = 1)
        )

    def forward(self, x):
        return x + self.block(x)    
