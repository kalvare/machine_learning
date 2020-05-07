"""
This file contains an implementation of the Cross Entropy loss function.
"""

import torch

from core.norms.l2_norm import l2_norm_model


def cross_entropy(log_probs, classes, reduce='mean'):
    """
    Computes the categorical cross-entropy loss.
    Args:
        log_probs: tensor, shape(n_samples, n_classes)
                   log probabilities over possible classes for each sample.
        classes:   tensor of int, shape(n_classes)
                   class labels. These should be integers in the range [0, n_classes - 1] and should index
                   log probabilities[i] appropriately.
        reduce:    str,
                   method for aggregating the loss over the samples. One of ('mean', 'sum').
    Returns:
        cross_entropy_loss: tensor,
                   the cross-entropy loss.

    """
    # Validate inputs
    implemented_reduction_choices = ('mean', 'sum')
    if reduce not in implemented_reduction_choices:
        raise ValueError('reduce must be one of %s, but %s was passed.' % (implemented_reduction_choices,
                                                                           reduce))
    n_classes = log_probs.shape[1]
    if classes.max() > n_classes or classes.min() < 0:
        raise ValueError('classes must be in range [0, n_classes - 1].')

    # Calculate cross entropy per sample
    n_samples = log_probs.shape[0]
    sample_cross_entropies = torch.tensor([-1 * log_probs[i][c] for i, c in zip(range(n_samples), classes)],
                                          device=log_probs.device)

    # Reduce as specified
    if reduce == 'mean':
        return torch.mean(sample_cross_entropies)
    elif reduce == 'sum':
        return torch.sum(sample_cross_entropies)


def penalized_cross_entropy(log_probs, classes, model_params, reduce='mean', weight_decay=0.0):
    """
    Computes the categorical cross-entropy loss, including l2 norm penalization for MAP estimation (gaussian prior).
    Computed as cross_entropy + (0.5) * weight_decay * l2_norm(model_parameters).
    Args:
        log_probs: tensor, shape(n_samples, n_classes)
                   log probabilities over possible classes for each sample.
        classes:   tensor of int, shape(n_classes)
                   class labels. These should be integers in the range [0, n_classes - 1] and should index
                   log probabilities[i] appropriately.
        model_params: tuple of tensor,
                   tuple of parameter sets for the model.
        reduce:    str,
                   method for aggregating the loss over the samples. One of ('mean', 'sum').
        weight_decay: float,
                   weight decay term (lambda) for controlling the contribution of the l2 norm of the network
                   parameters.
    Returns:
        cross_entropy_loss: tensor,
                   the cross-entropy loss.

    """

    # Calculate cross-entropy
    cross_entropy_unpenalized = cross_entropy(log_probs=log_probs, classes=classes, reduce=reduce)

    # Add l2 penalization
    return cross_entropy_unpenalized + (0.5 * weight_decay * l2_norm_model(model_params))
