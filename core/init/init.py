"""
This file contains functions for initializing neural network parameters.
"""

import torch
import numpy as np


# def kaiming_(x):
#     """
#     In-place Kaiming initialization.
#     Args:
#         x: Tensor,
#              Tensor to intialize. Values are changed in place.
#
#     Returns:
#         x_init: Tensor,
#              Reference to the modified tensor.
#
#     """
#     if not isinstance(x, torch.Tensor):
#         raise ValueError('X must be a Tensor but was type {}'.format(type(x)))
#
#     with torch.no_grad():
#         data = np.random.randn(x.shape) * np.sqrt(2 / x.shape[])
#         return x.fill_(data)

def xavier_uniform_fully_connected_(w):
    """
    In-place Kaiming initialization for the weight tensor for a fully connected layer.
    Specifically, it is assumed that w will be used in the equation: hW^T + b
    Args:
        w: Tensor,
             Tensor to initialize. Values are changed in place.

    Returns:
        w_init: Tensor,
             Reference to the modified tensor.

    """
    if not isinstance(w, torch.Tensor):
        raise ValueError('X must be a Tensor but was type {}'.format(type(x)))

    # Must be weight tensor for fully connected layer
    dim = w.dim()
    if dim != 2:
        raise ValueError('x must be a weight tensor for a fully connected layer.')

    with torch.no_grad():
        fan_in = w.size(1)
        fan_out = w.size(0)

        # Fill from U[-a, a] where a = sqrt(6) / sqrt(fan_in + fan_out)
        a = np.sqrt(6) / np.sqrt(fan_in + fan_out)
        return w.uniform_(-1 * a, a)
