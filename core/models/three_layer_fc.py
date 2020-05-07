"""
This file contains code for a 2 layer fully connected neural network for classification; written from scratch.
"""

import torch
import numpy as np

from core.activations import activations

class SimpleFullyConnected(object):
    """
    Simple 2 layer (single hidden) fully connected network for classification.
    This model uses a Cross-Entropy loss with L2 weight decay, relu activations for the hidden layer, and
    softmax activation for the output layer.
    """
    def __init__(self, in_features, hidden_size, n_classes, weight_init, dtype=torch.float,
                 device=torch.device('cpu')):
        """

        Args:
            in_features: int,
                         number of input features into the model.
            hidden_size: int,
                         number of neurons in the single hidden layer.
            n_classes:   int,
                         number of output classes to choose from.
            weight_init: str,
                         weight initialization strategy.
                         One of ('Kaiming')
            dtype: torch.dtype,
                         data type for model parameters.
            device: torch.device,
                         device to use for this model.
        """
        self.in_features = in_features
        self.hidden_size = hidden_size
        self.n_classes = n_classes
        self.dtype = dtype
        self.device = device

        # Model parameters
        self.W_1 = torch.empty(self.hidden_size, self.in_features, dtype=dtype, device=self.device)
        self.b_1 = torch.empty(self.hidden_size, dtype=dtype, device=self.device)
        self.W_2 = torch.empty(self.n_classes, self.hidden_size, dtype=dtype, device=self.device)
        self.b_2 = torch.empty(self.n_classes, dtype=dtype, device=self.device)
        self.initialize_params(weight_init=weight_init)

        # Gradients for model parameters
        self.gradient_b_2 = torch.zeros(self.b_2.size(), device=self.device)
        self.gradient_W_2 = torch.zeros(self.W_2.size(), device=self.device)
        self.gradient_b_1 = torch.zeros(self.b_1.size(), device=self.device)
        self.gradient_W_1 = torch.zeros(self.W_1.size(), device=self.device)

        # Computation graph
        # - forward
        self.batch_probabilities = None
        self.hidden = None
        self.hidden_pre_activation = None
        self.x = None
        # - backward
        self.gradient_unnormalized_log_probs = None
        self.gradient_hidden = None
        self.gradient_hidden_pre_activation = None



    def forward(self, x, return_type='probs'):
        """
        Forward propogation function.
        Args:
            x: Tensor of shape=(batch_size, n_features),
               data to forward propogate through the network.
            return_type: str,
               type of output desired. One of ('unnormalized_log_probs', 'log_probs', 'probs').

        Returns:
            depending on return_type, one of:
            probabilities: Tensor, shape=(batch_size, n_classes),
                                 Predicted probability distribution over the classes; one for each sample in the batch.
            unnormalized_log_probabilities: Tensor, shape(batch_size, n_classes),
                                 Predict unnormalized log probabilities over the classes; one for each sample in the
                                 batch.
            log_probabilities:   Predict log probabilities over the classes; one for each sample in the batch.


        """
        # Validate inputs
        implemented_return_types = ('unnormalized_log_probs', 'log_probs', 'probs')
        if return_type not in implemented_return_types:
            raise ValueError('valid return types are %s, but %s was passed.' % (implemented_return_types,
                                                                                return_type))
        self.x = x

        # Input to hidden
        self.hidden_pre_activation = torch.matmul(x, torch.transpose(self.W_1, dim0=0, dim1=1)) + self.b_1
        self.hidden = activations.relu(self.hidden_pre_activation)

        # Hidden to out
        unnormalized_log_probs = torch.matmul(self.hidden, torch.transpose(self.W_2, dim0=0, dim1=1)) + self.b_2
        log_probs = activations.log_softmax(unnormalized_log_probs)
        self.batch_probabilities = torch.exp(log_probs)

        if return_type == 'unnormalized_log_probs':
            return unnormalized_log_probs
        elif return_type == 'log_probs':
            return log_probs
        elif return_type == 'probs':
            return self.batch_probabilities

    def backward(self, ground_truth):
        """
        Backward pass using cross-entropy loss. This function sets the gradients for the model parameters.
        Args:
            ground_truth: tensor of int, shape=(batch_size)
                          ground truth labels.

        Returns:
            self.gradient_W_1,
            self.gradient_b_1,
            self.gradient_W_2,
            self.gradient_b_2

        """

        # Make sure the forward propogation has been run
        if not self._check_forward_ran():
            raise ValueError('Forward propogation must be run first.')

        batch_size = len(ground_truth)

        # Convert ground truth labels to one-hot-encoding
        y_one_hot = self._one_hot(ground_truth)

        # Gradient of the loss w.r.t unnormalized_log_probs for each sample in the batch - size(batch_size, n_classes)
        self.gradient_unnormalized_log_probs = self.batch_probabilities - y_one_hot

        # Gradient of the loss w.r.t b2 - shape(n_class) - this is the average over all samples in the batch
        self.gradient_b_2.add_(torch.mean(self.gradient_unnormalized_log_probs, dim=0))

        # Gradient of the loss w.r.t W2 - size(n_classes, n_hidden) - this is the average over all samples in the batch
        self.gradient_W_2.add_(torch.matmul(torch.transpose(self.gradient_unnormalized_log_probs, dim0=0, dim1=1),
                                         self.hidden) / batch_size)

        # Gradient of the loss w.r.t the hidden layer - size(batch_size, n_hidden)
        self.gradient_hidden = torch.matmul(self.gradient_unnormalized_log_probs, self.W_2)

        # Gradient of of the loss w.r.t the hidden layer pre-relu - size(batch_size, n_hidden)
        self.gradient_hidden_pre_activation = self.gradient_hidden * activations.relu_grad(self.hidden_pre_activation)

        # Gradient of the loss w.r.t b1 - size(n_hidden) - this is the average over all samples in the batch
        self.gradient_b_1.add_(torch.mean(self.gradient_hidden_pre_activation, dim=0))

        # Gradient of the loss w.r.t W1 - size(n_hidden, n_features) - this is the average over all samples in the batch
        self.gradient_W_1.add_(torch.matmul(torch.transpose(self.gradient_hidden_pre_activation, dim0=0, dim1=1),
                                         self.x) / batch_size)

        # Reset the cached variables in the computation graph
        self._reset_comp_graph_cache()

        return self.gradient_W_1, self.gradient_b_1, self.gradient_W_2, self.gradient_b_2

    def get_params(self):
        """
        Returns a a tuple of all parameter sets in the model.

        Returns: tuple of tensor,
                 tuple of all parameter sets in the model.

        """

        return self.W_1, self.b_1, self.W_2, self.b_2

    def get_gradients(self):
        """
        Returns a tuple of gradients for all parameter sets in the model.

        Returns: tuple of tensor,
                 tuple of gradients for all parameter sets in the model.
        """

        return self.gradient_W_1, self.gradient_b_1, self.gradient_W_2, self.gradient_b_2

    def _check_forward_ran(self):
        """
        Checks if the forward propogation has been run
        Returns: True or False

        """
        return self.batch_probabilities is not None and \
               self.hidden is not None and \
               self.hidden_pre_activation is not None and \
               self.x is not None

    def _reset_comp_graph_cache(self):
        """
        Resets the cache of computed variables in the computation graph from the forward pass
        Returns: None

        """
        self.batch_probabilities = None
        self.hidden = None
        self.hidden_pre_activation = None
        self.x = None

    def reset_gradients(self):
        self.gradient_unnormalized_log_probs = None
        self.gradient_b_2.copy_(torch.tensor(0))
        self.gradient_W_2.copy_(torch.tensor(0))
        self.gradient_hidden = None
        self.gradient_hidden_pre_activation = None
        self.gradient_b_1.copy_(torch.tensor(0))
        self.gradient_W_1.copy_(torch.tensor(0))


    def _one_hot(self, labels):
        """
        Converts a 1-D array of ints to one-hot-encoding. Labels are expected to be integers in the
        range [0, self.n_classes - 1]
        Args:
            labels: tensor of int, shape=(batch_size)
                    array to convert.

        Returns:
            one_hot: 2-D tensor of ints, shape=(batch_size, self.n_classes)

        """

        # Make sure all labels are in expected range
        if torch.max(labels) > (self.n_classes - 1) or torch.min(labels) < 0:
            raise ValueError('labels must be integers in the range [0, %d]' % (self.n_classes - 1))

        batch_size = len(labels)
        one_hot = torch.zeros(batch_size, self.n_classes, device=self.device)

        # Fill one-hot encoded matrix
        for i in range(batch_size):
            one_hot[i, labels[i]] = 1

        return one_hot


    def initialize_params(self, weight_init):
        """
        Initialize the parameters of the model in the specified way.
        Args:
            weight_init: str,
                         weight initialization strategy.
                         One of ('Kaiming')

        Returns: None

        """

        if weight_init not in ('Kaiming'):
            raise ValueError('Weight init must be \'Kaiming\'. You passed %s.' % weight_init)

        if weight_init == 'Kaiming':
            self.W_1 = torch.tensor(
                data=np.random.randn(self.W_1.shape[0], self.W_1.shape[1]) * np.sqrt(2 / self.in_features),
                requires_grad=False,
                dtype=self.dtype, device=self.device
            )
            self.W_2 = torch.tensor(
                data=np.random.randn(self.W_2.shape[0], self.W_2.shape[1]) * np.sqrt(2 / self.hidden_size),
                requires_grad=False,
                dtype=self.dtype, device=self.device
            )

        # Initialize bias to be small constant
        self.b_1 = torch.full(size=self.b_1.size(), fill_value=0.01, dtype=self.dtype, device=self.device)
        self.b_2 = torch.full(size=self.b_2.size(), fill_value=0.01, dtype=self.dtype, device=self.device)

    def to(self, device):
        """
        Moves this model to the specified device.
        Args:
            device: Device, Pytorch device to move the model to

        Returns: None

        """
        # Move params
        self.W_1 = self.W_1.to(device)
        self.W_2 = self.W_2.to(device)
        self.b_1 = self.b_1.to(device)
        self.b_2 = self.b_2.to(device)

        # Move gradients
        self.gradient_W_1 = self.gradient_W_1.to(device)
        self.gradient_W_2 = self.gradient_W_2.to(device)
        self.gradient_b_1 = self.gradient_b_1.to(device)
        self.gradient_b_2 = self.gradient_b_2.to(device)

        # Set model device
        self.device = device


