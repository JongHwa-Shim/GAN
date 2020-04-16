import torch.nn as nn
import torch

input = torch.randn(1,100,1,1)
#DCGAN generator convolution part
convtp1 = nn.ConvTranspose2d(in_channels=100, out_channels=28*8, 
                               kernel_size=7, stride=1, padding=0, 
                               bias=False)
convtp2 = nn.ConvTranspose2d(in_channels=28*8, out_channels=28*4, 
                               kernel_size=4, stride=2, padding=1, 
                               bias=False)

convtp3 = nn.ConvTranspose2d(in_channels=28*4, out_channels=1, 
                               kernel_size=4, stride=2, padding=1, 
                               bias=False)

input = convtp1(input) #output_size = [1,28*8,7,7]
input = convtp2(input) #output_size = [1,28*4,14,14]
input = convtp3(input) #output_size = [1,1,28,28]

output = convtp(input)
a = 0