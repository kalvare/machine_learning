"""
This file contains code for the implementation of Stochastic (mini-batch) Gradient Descent
"""

import torch


class SGD(object):
    """
    Class implementing Stochastic (mini-batch) Gradient Descent
    """

    def __init__(self, params, lr=0.001, weight_decay=0.0):
        """
        Constructor.
        Args:
            params: iterator of Parameter,
                    Iterator over Parameters to optimize.
            lr: float,
                    learning rate for numerical optimization.
            weight_decay: float,
                    weight decay for regularization.
        """

        # if not isinstance(params, torch.nn.Parameter):
        #     raise ValueError('params must be type torch.nn.Parameter but is type {}'.format(type(params)))

        self.params = list(params)
        self.lr = lr
        self.weight_decay = weight_decay

    @torch.no_grad()
    def step(self):
        """
        Takes a single optimization step.
        Returns: None

        """

        for param in self.params:
            grad = param.grad
            if self.weight_decay > 0:
                # Add weight decay directly to the gradients
                grad += self.weight_decay * param
            param.sub_(self.lr * grad)

    def zero_grad(self):
        """
        Zeros the gradients for all parameters known to this optimizer.
        Returns: None

        """

        for param in self.params:
            if param.grad is not None:
                param.grad.detach_()
                param.grad.zero_()