from common.autograd import Value
from model_implementation.layer.module import Module

class linear(Module):
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return x.linear()
    
    def parameters(self):
        return []

class relu(Module):
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return x.relu()
    
    def parameters(self):
        return []

class sigmoid(Module):
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return x.sigmoid()
    
    def parameters(self):
        return []

class tanh(Module):
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return x.tanh()
    
    def parameters(self):
        return []

class softmax(Module):
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return x.softmax()
    
    def parameters(self):
        return []

class leaky_relu(Module):
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return x.leaky_relu()
    
    def parameters(self):
        return []
class gelu(Module):
    def forward(self, x):
        x = x if isinstance(x, Value) else Value(x)
        return x.gelu()
    
    def parameters(self):
        return []
