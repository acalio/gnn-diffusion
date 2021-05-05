import torch as th
import functools


def weights_decorator(f):
    @functools.wraps(f)
    def wrapper(*args, **kwargs):
        try:
            w = kwargs['weights']
            if not w:
                kwargs['weights'] = th.ones_like(args[1])
        except KeyError:
            if len(args) <= 3:
                kwargs['weights'] = th.ones_like(args[1])
            elif not args[-1]:
                args = (*args[:-1], th.ones_like(args[1]))
        
        return f(*args, **kwargs)
    return wrapper


class Loss:
    def __init__(self, reduction):
        self.reduction = {
            "sum": th.sum,
            "mean": th.mean
        }[reduction]

    def __call__(self, pred, labels, weights=None):
        raise NotImplementedError

class LogCosh(Loss):
    def __init__(self, reduction):
        super(LogCosh, self).__init__(reduction)

    @weights_decorator
    def __call__(self, pred, labels, weights=None):
        return self.reduction(th.log(weights * th.cosh(pred - labels)))

class KL(Loss):
    def __init__(self, reduction):
        super(KL, self).__init__(reduction)

    def __call__(self, pred, labels, weights=None):
        mask = (pred > 0) & (labels > 0)
        return self.reduction(th.pow(pred[~mask] - labels[~mask], 3)) +\
            self.reduction(pred[mask]*(th.log(pred[mask])-th.log(labels[mask])))
    

class MSE(Loss):
    def __init__(self, reduction):
        super(MSE, self).__init__(reduction)

    @weights_decorator
    def __call__(self, pred, labels, weights=None):
        return self.reduction(weights * ((pred-labels)**2))


class MAE(Loss):
    def __init__(self, reduction):
        super(MAE, self).__init__(reduction)

    @weights_decorator
    def __call__(self, pred, labels, weights=None):
        return self.reduction(weights * th.abs(pred-labels))


class Huber(Loss):
    def __init__(self, reduction, beta=1):
        super(Huber, self).__init__(reduction)
        self.beta = beta

    @weights_decorator
    def __call__(self, pred, labels, weights=None):
        absolute_error = th.abs(pred - labels)
        error = th.where(absolute_error < self.beta, .5*(absolute_error**2)/self.beta, absolute_error-.5*self.beta)
        return self.reduction(weights*error)



    

        
