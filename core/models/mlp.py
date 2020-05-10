import torch

from core.modules.linear import Affine


class MLP(torch.nn.Module):

    def __init__(self, in_features, hidden_sizes, hidden_activation, n_out, init_w, init_b):
        """
        A Multi Layer Perceptron.
        Args:
            in_features: int,
                 Number of input features.
            hidden_sizes: list-like of int,
                 Size of each hidden layer (number of neurons).
            hidden_activation: callable,
                 Activation function to apply to the output of each hidden layer.
            n_out: int,
                 Number of outputs to produce. For example, if this is a classification problem, this should be
                 the number of classes.
            init_w: callable,
                 Initialization function to be used for all weight matrices.
            init_b: callable,
                 Initialization function to be used for all bias vectors.
        """
        super(MLP, self).__init__()
        self.in_features = in_features
        self.hidden_sizes = hidden_sizes
        self.n_out = n_out
        self.init_w = init_w
        self.init_b = init_b
        self.hidden_activation = hidden_activation

        # Create the layers for this model
        n_layers = len(hidden_sizes)
        layers = []
        # Add hidden layers
        for i in range(n_layers):
            layers.append(Affine(in_features=in_features, out_features=hidden_sizes[i], weight_init_w=self.init_w,
                                 weight_init_b=self.init_b, bias=True))
            in_features = hidden_sizes[i]
        # Add output layer
        layers.append(Affine(in_features=in_features, out_features=self.n_out, weight_init_w=self.init_w,
                                 weight_init_b=self.init_b, bias=True))
        # Register layers with Pytorch
        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x):
        """
        Forward propagation.
        Args:
            x: tensor, shape(batch_size, in_features),
                 Batch of data to run through the network.

        Returns:
            out: tensor, shape(batch_size, n_out),
                 Network outputs. No activation is performed on the outputs of the last Affine layer.


        """

        # Forward the batch of data through all modules in the network and apply the activation functions
        for module in self.layers[:-1]:
            # Affine transformation
            x = module(x)
            # Activation
            x = self.hidden_activation(x)

        # Return the affine transformation from the output layer
        return self.layers[-1](x)


