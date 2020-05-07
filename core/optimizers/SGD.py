"""
This file contains code for the implementation of Stochastic (mini-batch) Gradient Descent
"""

class SGD(object):
    """
    Class implementing Stochastic (mini-batch) Gradient Descent
    """

    def __init__(self, params, gradients, lr=0.001, weight_decay=0.0):
        """
        Constructor.
        Args:
            params: tuple of tensor,
                    parameters to optimize. This should be a tuple of all parameter sets in the model to update.
                    Each parameter set should be a tensor.
            gradients: tuple of tensor,
                    corresponding gradients for params. Again, this should be a tuple with a one-to-one
                    correspondence to params.
            lr: float,
                    learning rate for numerical optimization.
            weight_decay: float,
                    weight decay for regularization.
        """

        self.params = params
        self.gradients = gradients
        self.lr = lr
        self.weight_decay = weight_decay

    def step(self):
        """
        Takes a single optimization step.
        Returns: None

        """

        for params, gradients in zip(self.params, self.gradients):
            grad = gradients
            if self.weight_decay > 0:
                # Add weight decay directly to the gradients
                grad += self.weight_decay * params
            params.sub_(self.lr * grad)
