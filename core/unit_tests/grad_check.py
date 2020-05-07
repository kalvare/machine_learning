"""
This file contains code for implementing gradient checking
"""

import torch
import numpy as np

from core.models.three_layer_fc import SimpleFullyConnected


def gradient_check(x, y, model, parameter, backward_grad_idx, cost_fn, kwargs):
    """
    Performs gradient check by evaluating the analytical gradient using the backward()
    function of the model, and comparing this with approximated numerical gradients.
    Args:
        x: tensor, shape(batch_size, n_features)
           Input for the model.
        y: tensor of ints, shape(batch_size)
           Output labels for the input.
        model: differential function,
           The model to check the gradients for. Must implement forward() and backward() functions.
        parameter: tensor, shape(1)
           Single parameter in the model. Must be a tensor for in place modifications. Value will be modifed
           and then restored to initial value.
        backward_grad_idx: tuple of int,
           Indices to be applied to the output of backward() to reach the gradient of parameter. They are applied
           to the output recursively.
        cost_fn: callable,
           Cost function that the analytical gradients computed during backward() are taken w.r.t
        kwargs: dict,
           Key word arguments for the cost function

    Returns:
        absolute_error: # TODO: doc
        relative_error:

    """

    # Forward pass
    out = model.forward(x)

    # Compute analytical gradients
    model.reset_gradients()
    grads = model.backward(ground_truth=y)

    # Get the analytical gradient for the parameter being modified
    idx_layer = 0
    analytical_grad_param = grads[backward_grad_idx[idx_layer]]
    idx_layer += 1
    while idx_layer < len(backward_grad_idx):
        analytical_grad_param = analytical_grad_param[backward_grad_idx[idx_layer]]
        idx_layer += 1

    # Modify parameter by small amount and re-evalutate the function
    param_init_value = parameter.item() # To restore later
    h = 1e-5

    # Compute f(x + h)
    parameter.add_(h)
    _, f_x_plus_h = model.forward(x, return_unnorm_log_probs=True)
    f_x_plus_h = cost_fn(f_x_plus_h, y, **kwargs)

    # Compute f(x - h)
    parameter.sub_(2 * h)
    _, f_x_sub_h = model.forward(x, return_unnorm_log_probs=True)
    f_x_sub_h = cost_fn(f_x_sub_h, y, **kwargs)

    # Restore original parameter value
    parameter.copy_(torch.tensor(param_init_value))

    # Approximate numerical gradient as (f(x + h) - f(x - h)) / (2 * h)
    numerical_gradient_param = (f_x_plus_h - f_x_sub_h) / (2 * h)

    # Calculate absolute and relative errors
    abs_error = abs(analytical_grad_param - numerical_gradient_param)
    relative_error = abs(analytical_grad_param - numerical_gradient_param) / \
                     max(abs(analytical_grad_param), abs(numerical_gradient_param))

    model.reset_gradients()

    return abs_error, relative_error
