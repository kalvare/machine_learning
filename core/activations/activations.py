"""
This file contains element-wise activation functions, and their gradients.
"""

import torch

def relu(x):
    """
    ReLU element-wise activation.
    Args:
        x: array-like, any shape,
           array to compute activations for.

    Returns:
        x_relu: same type and shape as x,
                element wise application of ReLU to x.

    """

    return x * (x > 0)


def relu_grad(x):
    """
    Gradient of the ReLU element-wise activation evaluated at x.
    Args:
        x: tensor,
           Each element of x is a point for which the gradient of ReLU is evaluated.

    Returns:

    """

    return (x > 0).float()

def log_softmax(x):
    """
    Computes the log-softmax activation over the last dimension of x.
    Args:
        x: Tensor,
           data to compute the softmax activations for.

    Returns:
        log_x_softmax, Tensor same shape as x.
                       log-softmax activations of x over the last dimension.

    """

    log_denom = torch.logsumexp(x, dim=(len(x.shape) - 1), keepdim=True)
    log_softmax = x - log_denom

    return log_softmax

def softmax(x):
    """
    Computes the softmax activation over the last dimension of x.
    Args:
        x: Tensor,
           data to compute the softmax activations for.

    Returns:
        x_softmax, Tensor same shape as x.
                   softmax activations of x over the last dimension.

    """

    # First compute the log-softmax
    log_softmax_ = log_softmax(x)

    # Then exponentiate
    x_softmax = torch.exp(log_softmax_)
    return x_softmax
