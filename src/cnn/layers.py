import numpy as np
from model_implementation.layer.module import Module
from common.autograd.autograd import Value

class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        
        self.weight = None # Shape: (kH, kW, C_in, C_out)
        self.bias = None   # Shape: (C_out,)

    def forward(self, x):
        # x shape: (N, H, W, C_in)
        if isinstance(x, Value): x = x.data
        
        N, H, W, C_in = x.shape
        kH, kW = self.kernel_size
        out_H = (H + 2*self.padding - kH) // self.stride + 1
        out_W = (W + 2*self.padding - kW) // self.stride + 1
  
        if self.padding > 0:
            x = np.pad(x, ((0,0), (self.padding, self.padding), (self.padding, self.padding), (0,0)), mode='constant')

        output = np.zeros((N, out_H, out_W, self.weight.shape[-1]))
        
        for i in range(out_H):
            for j in range(out_W):
                h_start, w_start = i * self.stride, j * self.stride
                patch = x[:, h_start:h_start+kH, w_start:w_start+kW, :] # (N, kH, kW, C_in)
                output[:, i, j, :] = np.tensordot(patch, self.weight, axes=((1, 2, 3), (0, 1, 2))) + self.bias
                
        return output

class MaxPooling2D(Module):
    def __init__(self, pool_size=(2, 2), stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        N, H, W, C = x.shape
        kH, kW = self.pool_size
        out_H = (H - kH) // self.stride + 1
        out_W = (W - kW) // self.stride + 1
        
        output = np.zeros((N, out_H, out_W, C))
        for i in range(out_H):
            for j in range(out_W):
                h_s, w_s = i * self.stride, j * self.stride
                output[:, i, j, :] = np.max(x[:, h_s:h_s+kH, w_s:w_s+kW, :], axis=(1, 2))
        return output

class Flatten(Module):
    def forward(self, x):
        return x.reshape(x.shape[0], -1)