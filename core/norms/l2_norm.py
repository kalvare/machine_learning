"""
This file contains an implementation of the l2 norm for tensors and network parameters.
"""

def l2_norm(x):
    """
    Calculates the l2 norm of the input tensor.
    Args:
        x: tensor, any shape
           tensor to take the l2 norm of.

    Returns: float,
           l2 norm of the input

    """

    return x.view(-1).dot(x.view(-1))

def l2_norm_model(net_params):
    """
    Calculates the l2 norm of all parameters in a model.
    Args:
        net_params: tuple of tensor,
                    tuple of parameter sets for a model.

    Returns: float,
             l2 norm of all parameters in a model.

    """

    norm = 0
    for param_set in net_params:
        norm += l2_norm(param_set)

    return norm
