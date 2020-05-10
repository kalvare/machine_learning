"""
Contains code for linear modules
"""

import torch
from torch.nn.parameter import Parameter


class Affine(torch.nn.Module):
    def __init__(self, in_features, out_features, weight_init_w, weight_init_b, bias=True):
        """
        Computes an affine transformation as xW^T + b.
        Args:
            in_features: int,
                 The number of input features for a single sample.
            out_features: int,
                 The number of output features to produce for a single sample.
            weight_init_w: callable,
                 The weight initialization method for the weight matrix. This method should accept a tensor and fill
                 that tensor in place.
            weight_init_b: callable,
                 The weight initialization method for the bias vector. This method should accept a tensor and fill
                 that tensor in place.
            bias: bool,
                 Whether to use a bias term or not.
        """
        super(Affine, self).__init__()

        self.in_features = in_features
        self.out_features = out_features

        # Initialize empty W matrix and register as parameter
        self.W = Parameter(torch.Tensor(self.out_features, self.in_features))
        # Initialize empty bias vector and register as parameter, or register the bias as not existing
        if bias:
            self.bias = Parameter(torch.Tensor(self.out_features))
        # else:
        #     self.register_parameter('bias', None) #TODO: Remove if not needed

        # Initialize all parameters using the initialization function
        weight_init_w(self.W)
        if bias:
            weight_init_b(self.bias)
        else:
            self.bias = 0

    def forward(self, x):
        """
        Forward propagation for this module.
        Args:
            x: Tensor, shape(batch_size, in_features)
                 Input tensor.

        Returns:
            out: Tensor, shape(batch_size, out_features)
                 Output tensor.
        """

        return torch.matmul(x, torch.transpose(self.W, dim0=0, dim1=1)) + self.bias



