import numpy as np
from model_implementation.layer.module import Module
from common.autograd import Value

# Shared Parameter
class Conv2D(Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.padding = padding
        self.weight = None # (kH, kW, C_in, C_out)
        self.bias = None   # (C_out,)

    def forward(self, x):
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
                patch = x[:, h_start:h_start+kH, w_start:w_start+kW, :]
                output[:, i, j, :] = np.tensordot(patch, self.weight, axes=((1, 2, 3), (0, 1, 2))) + self.bias
                
        return output

# Non-shared Parameter
class LocallyConnected2D(Module):
    def __init__(self, in_channels, out_channels, input_size, kernel_size, stride=1):
        super().__init__()
        self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        self.stride = stride
        self.input_size = input_size
        self.weight = None 
        self.bias = None 

    def forward(self, x):
        if isinstance(x, Value): x = x.data

        N, H, W, C_in = x.shape
        kH, kW = self.kernel_size
        out_H = (H + 2 * self.padding - kH) // self.stride + 1
        out_W = (W + 2 * self.padding - kW) // self.stride + 1
        C_out = self.bias.shape[-1]

        if self.padding > 0:
            x = np.pad(
                x,
                ((0, 0), (self.padding, self.padding), (self.padding, self.padding), (0, 0)),
                mode="constant",
            )

        output = np.zeros((N, out_H, out_W, C_out))
        for i in range(out_H):
            for j in range(out_W):
                h_s, w_s = i * self.stride, j * self.stride
                patch = x[:, h_s:h_s+kH, w_s:w_s+kW, :].reshape(N, -1)
                idx = i * out_W + j
                output[:, i, j, :] = (patch @ self.weight[idx]) + self.bias[i, j]
        return apply_activation(output, getattr(self, "activation", None))

# Sliding window pooling
class MaxPooling2D(Module):
    def __init__(self, pool_size=(2, 2), stride=2):
        super().__init__()
        self.pool_size = pool_size
        self.stride = stride

    def forward(self, x):
        if isinstance(x, Value): x = x.data
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

class AveragePooling2D(Module):
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
                output[:, i, j, :] = np.mean(x[:, h_s:h_s+kH, w_s:w_s+kW, :], axis=(1, 2))
        return output

# Global Pooling
class GlobalAveragePooling2D(Module):
    def forward(self, x):
        return np.mean(x, axis=(1, 2))
    
class GlobalMaxPooling2D(Module):
    def forward(self, x):
        return np.max(x, axis=(1, 2))

# Flatten
class Flatten(Module):
    def forward(self, x):
        if isinstance(x, Value): x = x.data
        return x.reshape(x.shape[0], -1)

class Dense(Module):
    def __init__(self, units=None, activation=None):
        super().__init__()
        self.units = units
        self.activation = activation
        self.weight = None
        self.bias = None

    def load_keras(self, weights):
        if len(weights) < 2:
            raise ValueError("Dense layer expects kernel and bias weights.")
        self.weight = np.asarray(weights[0], dtype="float32")
        self.bias = np.asarray(weights[1], dtype="float32")
        if self.units is None:
            self.units = int(self.bias.shape[0])
        return self

    def forward(self, x):
        if isinstance(x, Value):
            x = x.data
        output = np.asarray(x) @ self.weight + self.bias
        return apply_activation(output, self.activation)

class ReLU(Module):
    def forward(self, x):
        if isinstance(x, Value):
            x = x.data
        return np.maximum(x, 0)

class Softmax(Module):
    def forward(self, x):
        if isinstance(x, Value):
            x = x.data
        return softmax(np.asarray(x))

class Sequential(Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = list(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def predict(self, x):
        return self.forward(x)

    def count_params(self):
        total = 0
        for layer in self.layers:
            for name in ("weight", "bias"):
                value = getattr(layer, name, None)
                if value is not None:
                    total += int(np.asarray(value).size)
        return total

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / np.sum(exp, axis=-1, keepdims=True)

def apply_activation(x, activation):
    if activation in (None, "linear"):
        return x
    name = str(activation).lower()
    if name == "relu":
        return np.maximum(x, 0)
    if name == "softmax":
        return softmax(x)
    raise ValueError(f"Unsupported activation: {activation}")

def _normal_stride(value):
    if isinstance(value, tuple):
        return int(value[0])
    return int(value)

def _conv_init(
    self,
    in_channels=None,
    out_channels=None,
    kernel_size=(3, 3),
    stride=1,
    padding=0,
    filters=None,
    strides=None,
    activation=None,
):
    Module.__init__(self)
    self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    self.stride = _normal_stride(strides if strides is not None else stride)
    self.padding = int(padding)
    self.activation = activation
    self.weight = None
    self.bias = None
    self.in_channels = in_channels
    self.out_channels = filters if filters is not None else out_channels

def _local_init(
    self,
    in_channels=None,
    out_channels=None,
    input_size=None,
    kernel_size=(3, 3),
    stride=1,
    padding=0,
    filters=None,
    strides=None,
    activation=None,
):
    Module.__init__(self)
    self.kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
    self.stride = _normal_stride(strides if strides is not None else stride)
    self.padding = int(padding)
    self.input_size = input_size
    self.activation = activation
    self.weight = None
    self.bias = None
    self.in_channels = in_channels
    self.out_channels = filters if filters is not None else out_channels

def _pool_init(self, pool_size=(2, 2), stride=2, strides=None):
    Module.__init__(self)
    self.pool_size = pool_size if isinstance(pool_size, tuple) else (pool_size, pool_size)
    self.stride = _normal_stride(strides if strides is not None else stride)

def _set_common(layer, kernel, bias, activation=None):
    layer.weight = np.asarray(kernel, dtype="float32")
    layer.bias = np.asarray(bias, dtype="float32")
    layer.activation = activation
    return layer

def _conv_load_keras(self, weights):
    if len(weights) < 2:
        raise ValueError("Conv2D layer expects kernel and bias weights.")
    return _set_common(self, weights[0], weights[1], getattr(self, "activation", None))

def _conv_forward_with_activation(self, x):
    output = _conv_original_forward(self, x)
    return apply_activation(output, getattr(self, "activation", None))

def _local_load_keras(self, weights):
    if len(weights) < 2:
        raise ValueError("LocallyConnected2D layer expects kernel and bias weights.")
    kernel = np.asarray(weights[0], dtype="float32")
    bias = np.asarray(weights[1], dtype="float32")
    self.weight = kernel
    if bias.ndim == 2:
        out_h = (self.input_size[0] - self.kernel_size[0]) // self.stride + 1
        out_w = (self.input_size[1] - self.kernel_size[1]) // self.stride + 1
        bias = bias.reshape(out_h, out_w, bias.shape[-1])
    self.bias = bias
    return self

def _local_load_tiled_conv(self, kernel, bias, output_size):
    out_h, out_w = output_size
    flat_kernel = np.asarray(kernel, dtype="float32").reshape(-1, kernel.shape[-1])
    self.weight = np.tile(flat_kernel[None, :, :], (out_h * out_w, 1, 1))
    self.bias = np.tile(np.asarray(bias, dtype="float32")[None, None, :], (out_h, out_w, 1))
    return self

_conv_original_forward = Conv2D.forward
Conv2D.__init__ = _conv_init
Conv2D.load_keras = _conv_load_keras
Conv2D.forward = _conv_forward_with_activation
LocallyConnected2D.__init__ = _local_init
LocallyConnected2D.load_keras = _local_load_keras
LocallyConnected2D.load_tiled_conv = _local_load_tiled_conv
MaxPooling2D.__init__ = _pool_init
AveragePooling2D.__init__ = _pool_init

Conv2DLayer = Conv2D
LocallyConnected2DLayer = LocallyConnected2D
MaxPooling2DLayer = MaxPooling2D
AveragePooling2DLayer = AveragePooling2D
GlobalAveragePooling2DLayer = GlobalAveragePooling2D
GlobalMaxPooling2DLayer = GlobalMaxPooling2D
FlattenLayer = Flatten
DenseLayer = Dense
