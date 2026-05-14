import math
import numpy as np

# ------------------------------------------------------------------------------------------------------------
# biar pas gradien ngga aneh-aneh, bias dipisahin dari matrix (wx+b di slide kuliah)
# untuk operasi yang melibatkan bias, harus ada cara untuk ngotak-ngatik biar ukuran bias sama dengan matrix
def unbroadcast(grad, shape):
    while len(grad.shape) > len(shape):
        grad = grad.sum(axis=0)
    for i, dim in enumerate(shape):
        if dim == 1:
            grad = grad.sum(axis=i, keepdims=True)
    return grad
# ------------------------------------------------------------------------------------------------------------

class Value:
    def __init__(self, data, _children=(), _op='', label=''):
        self.data = np.asarray(data, dtype=float)
        self.grad = np.zeros_like(self.data)
        self._backward = lambda: None
        self._prev = set(_children)
        self._op = _op
        self.label = label # for clarity purposes

    def __repr__(self):
        return f"Value(data={self.data})"
    
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')

        def _backward():
            self.grad += unbroadcast(out.grad, self.data.shape)
            other.grad += unbroadcast(out.grad, other.data.shape)
        out._backward = _backward
        return out

    # cool stuff dari karpathy, siapa tau kepake
    def __radd__(self, other):
        return self + other
    
    def __neg__(self):
        return self * -1
    
    def __sub__(self, other):
        return self + (-other)
    
    def __rsub__(self, other):
        return other + (-self)

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')

        def _backward():
            self.grad += unbroadcast(other.data * out.grad, self.data.shape)
            other.grad += unbroadcast(self.data * out.grad, other.data.shape)
        out._backward = _backward
        return out
    
    # more cool stuff
    def __rmul__(self, other):
        return self * other
    
    def __truediv__(self, other):
        return self * (other**-1)

    # bedakan @ dengan *. Di numpy maknanya beda :D
    def __matmul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        assert self.data.ndim == 2
        assert other.data.ndim == 2

        out = Value(self.data @ other.data, (self, other), '@')

        def _backward():
            self.grad += out.grad @ other.data.T
            other.grad += self.data.T @ out.grad

        out._backward = _backward
        return out

    def sum(self, axis=None, keepdims=False):
        out = Value(
            self.data.sum(axis=axis, keepdims=keepdims),
            (self,),
            'sum'
        )

        def _backward():
            grad = out.grad

            if axis is not None and not keepdims:
                grad = np.expand_dims(grad, axis=axis)

            self.grad += np.ones_like(self.data) * grad

        out._backward = _backward
        return out
    
    def mean(self, axis=None, keepdims=False):
        divisor = self.data.size if axis is None else self.data.shape[axis]
        return self.sum(axis=axis, keepdims=keepdims) * (1.0 / divisor)
    
    def reshape(self, *shape):
        out = Value(self.data.reshape(*shape), (self,), 'reshape')

        def _backward():
            self.grad += out.grad.reshape(self.data.shape)

        out._backward = _backward
        return out

    @property
    def T(self):
        out = Value(self.data.T, (self,), 'transpose')

        def _backward():
            self.grad += out.grad.T

        out._backward = _backward
        return out
    
    def __getitem__(self, idx):
        out = Value(self.data[idx], (self,), 'slice')

        def _backward():
            grad = np.zeros_like(self.data)
            grad[idx] += out.grad
            self.grad += grad

        out._backward = _backward
        return out

    @staticmethod
    def cat(values, axis=1):
        data = np.concatenate([v.data for v in values], axis=axis)
        out = Value(data, values, 'cat')

        def _backward():
            start = 0
            for v in values:
                size = v.data.shape[axis]

                slicer = [slice(None)] * out.grad.ndim
                slicer[axis] = slice(start, start + size)

                v.grad += out.grad[tuple(slicer)]
                start += size

        out._backward = _backward
        return out

    @staticmethod
    def stack(values, axis=0):
        data = np.stack([v.data for v in values], axis=axis)
        out = Value(data, values, 'stack')

        def _backward():
            for i, v in enumerate(values):
                slicer = [slice(None)] * out.grad.ndim
                slicer[axis] = i
                v.grad += out.grad[tuple(slicer)]

        out._backward = _backward
        return out

    def log(self):
        out = Value(np.log(self.data), (self,), 'log')

        def _backward():
            self.grad += (1 / self.data) * out.grad

        out._backward = _backward
        return out

    def exp(self):
        x = self.data
        out = Value(np.exp(x), (self, ), 'exp')

        def _backward():
            self.grad += out.data * out.grad
        out._backward = _backward

        return out
    
    def __pow__(self, other):
        out = Value(self.data**other, (self, ), f'**{other}')
        
        def _backward():
            self.grad += other * (self.data**(other-1)) * out.grad
        out._backward = _backward

        return out
    
    # ------------------------------------------------------------
    # ACTIVATION FUNCTIONS
    # ------------------------------------------------------------
    def linear(self):
        x = self.data
        t = x
        out = Value(t, (self, ), 'linear')
        
        def _backward():
            self.grad += out.grad
        out._backward = _backward

        return out

    def relu(self):
        x = self.data
        t = np.maximum(0, x)
        out = Value(t, (self, ), 'relu')

        def _backward():
            self.grad += (x>0) * out.grad
        out._backward = _backward

        return out

    def sigmoid(self):
        x = self.data
        t = 1/(1+np.exp(-x))
        out = Value(t, (self, ), 'sigmoid')

        def _backward():
            self.grad += t*(1-t) * out.grad
        out._backward = _backward

        return out

    def tanh(self):
        x = self.data
        t = np.tanh(x)
        out = Value(t, (self, ), 'tanh')

        def _backward():
            self.grad += (1-t**2) * out.grad
        out._backward = _backward

        return out
    
    def softmax(self, axis=1):
        x = self.data
        shift = x - np.max(x, axis=axis, keepdims=True) # biar ngga overflow
        exp = np.exp(shift)
        t = exp / np.sum(exp, axis=axis, keepdims=True)
        out = Value(t, (self, ), 'softmax')

        def _backward():
            grad = out.grad
            s = out.data
            dot = np.sum(grad*s, axis=axis, keepdims=True)
            dx = s * (grad - dot)
            self.grad += dx
        out._backward = _backward

        return out
    
    def leaky_relu(self, alpha=0.01):
        x = self.data
        t = np.where(x > 0, x, alpha * x) # numpy magic
        out = Value(t, (self, ), 'leaky_relu')

        def _backward():
            dx = np.where(x > 0, 1, alpha)
            self.grad += dx * out.grad
        out._backward = _backward

        return out

    # contekan turunan: https://arxiv.org/pdf/2305.12073
    def gelu(self):
        x = self.data
        t = 0.5 * x * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3)))
        out = Value(t, (self, ), 'gelu')

        def _backward():
            grad = 0.5 * (1 + np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))) + 0.5 * x * (1 - np.tanh(np.sqrt(2/np.pi) * (x + 0.044715 * x**3))**2) * np.sqrt(2/np.pi) * (1 + 3 * 0.044715 * x**2)
            self.grad += grad * out.grad
        out._backward = _backward

        return out
        
    # ------------------------------------------------------------
    # BACKPROP MAGIC FROM KARPATHY
    # ------------------------------------------------------------
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)
        self.grad = np.ones_like(self.data)
        for node in reversed(topo):
            node._backward()
    
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)

if __name__ == "__main__":
    a = Value([1,2])
    b = [3,4]
    c = 10 + a + b
    print(c)
    print(c._prev)
    print(c._op)
