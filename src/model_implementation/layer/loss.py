from common.autograd import Value
import numpy as np

# base class
class Loss:
    def forward(self, prediction, target):
        raise NotImplementedError

    def __call__(self, prediction, target):
        return self.forward(prediction, target)

class MSELoss(Loss):
    def forward(self, prediction, target):
        prediction = prediction if isinstance(prediction, Value) else Value(prediction)
        target = target if isinstance(target, Value) else Value(target)

        diff = prediction.data - target.data
        loss = np.mean(diff**2)

        out = Value(loss, (prediction, target), 'mse')
        def _backward():
            grad = 2*diff/diff.size
            prediction.grad += grad*out.grad
            target.grad += -grad*out.grad
        out._backward = _backward

        return out

class BCELoss(Loss):
    def forward(self, prediction, target):
        prediction = prediction if isinstance(prediction, Value) else Value(prediction)
        target = target if isinstance(target, Value) else Value(target)

        if target.data.shape != prediction.data.shape:
            if (
                target.data.ndim == 1
                and prediction.data.ndim == 2
                and prediction.data.shape[1] == 1
                and target.data.shape[0] == prediction.data.shape[0]
            ):
                target = Value(target.data.reshape(-1, 1), label=target.label)

        eps = 1e-12 # biar ngga kena log(0) atau edge case lainnya
        p = np.clip(prediction.data, eps, 1 - eps)
        y = target.data

        loss = -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)) #np.log() udah basis e, bukan 10 ataupun 2...
        out = Value(loss, (prediction, target), 'bce')

        def _backward():
            grad = ((-y/p) + (1-y)/(1-p)) / y.size
            prediction.grad += grad * out.grad
        out._backward = _backward

        return out

class CCELoss(Loss):
    def forward(self, prediction, target):
        prediction = prediction if isinstance(prediction, Value) else Value(prediction)
        target = target if isinstance(target, Value) else Value(target)
        
        eps = 1e-12
        p = np.clip(prediction.data, eps, 1 - eps) # alasan idem bce
        y = target.data

        loss = -np.mean(np.sum(y*np.log(p), axis=1))
        out = Value(loss, (prediction, target), 'cce')

        def _backward():
            grad = -y/p
            grad /= y.shape[0]
            prediction.grad += grad * out.grad
        out._backward = _backward

        return out
