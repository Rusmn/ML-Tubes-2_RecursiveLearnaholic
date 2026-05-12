import numpy as np

class Optimizer:
    def __init__(self, params, lr=0.001, name=""):
        self.params = list(params)
        self.lr = lr
        self.name = name or self.__class__.__name__
        self.step_count = 0

    def step(self):
        raise NotImplementedError

    def zero_grad(self):
        for param in self.params:
            if hasattr(param, "zero_grad"):
                param.zero_grad()

class SGD(Optimizer):
    def __init__(self, params, lr=0.01, momentum=0.0, weight_decay=0.0, name=""):
        super().__init__(params=params, lr=lr, name=name or "SGD")
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.velocity = {}

    def step(self):
        self.step_count += 1
        for param in self.params:
            if param is None:
                continue

            grad = np.array(param.grad, copy=False)
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data

            if self.momentum:
                velocity = self.velocity.get(param)
                if velocity is None:
                    velocity = np.zeros_like(param.data)
                velocity = self.momentum * velocity - self.lr * grad
                self.velocity[param] = velocity
                param.data += velocity
            else:
                param.data -= self.lr * grad


class Adam(Optimizer):
    def __init__(self, params, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.0, name=""):
        super().__init__(params=params, lr=lr, name=name or "Adam")
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.weight_decay = weight_decay
        self.m = {}
        self.v = {}

    def step(self):
        self.step_count += 1
        for param in self.params:
            if param is None:
                continue

            grad = np.array(param.grad, copy=False)
            if self.weight_decay:
                grad = grad + self.weight_decay * param.data

            m = self.m.get(param)
            v = self.v.get(param)
            if m is None:
                m = np.zeros_like(param.data)
            if v is None:
                v = np.zeros_like(param.data)

            m = self.beta1 * m + (1 - self.beta1) * grad
            v = self.beta2 * v + (1 - self.beta2) * (grad * grad)
            self.m[param] = m
            self.v[param] = v

            m_hat = m / (1 - self.beta1 ** self.step_count)
            v_hat = v / (1 - self.beta2 ** self.step_count)
            param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
