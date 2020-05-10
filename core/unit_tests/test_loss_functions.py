import unittest
import torch
import numpy as np
import torchvision
from torch.nn.functional import cross_entropy as pytorch_cross_entropy

from core.loss_functions.cross_entropy import cross_entropy
from core.models.three_layer_fc import SimpleFullyConnected
from core.optimizers.SGD_v1 import SGD

class TestCrossEntropy(unittest.TestCase):

    def test_cross_entropy(self):
        device = torch.device('cpu')

        ### Train on MNIST using mini-batch SGD ###
        # Build model
        n_classes = 10
        n_features = 28 * 28
        net = SimpleFullyConnected(in_features=n_features, hidden_size=150, n_classes=n_classes,
                                   weight_decay=0, weight_init='Kaiming')
        net.to(device)

        # Build optimizer
        optim = SGD(params=net.get_params(), gradients=net.get_gradients(), lr=0.001, weight_decay=0.005)

        batch_size = 128
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('data', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size, shuffle=True, drop_last=True)
        batch_generator = iter(train_loader)

        # Test that the cross_entropy implementation returns the same results as the Pytorch one
        for i in range(800):
            print(i)
            try:
                imgs, labels = next(batch_generator)
            except StopIteration:
                batch_generator = iter(train_loader)
                imgs, labels = next(batch_generator)

            imgs = imgs.view(batch_size, -1)
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass
            unnormalized_log_probs = net.forward(imgs, return_type='unnormalized_log_probs')
            log_probs = net.forward(imgs, return_type='log_probs')

            loss_pytorch_mean = pytorch_cross_entropy(input=unnormalized_log_probs,
                                                             target=labels, reduction='mean')
            loss_implemented_mean = cross_entropy(log_probs=log_probs, classes=labels, reduce='mean')

            loss_pytorch_sum = pytorch_cross_entropy(input=unnormalized_log_probs,
                                                      target=labels, reduction='sum')
            loss_implemented_sum = cross_entropy(log_probs=log_probs, classes=labels, reduce='sum')

            self.assertTrue(np.isclose(loss_pytorch_mean, loss_implemented_mean))
            self.assertTrue(np.isclose(loss_pytorch_sum, loss_implemented_sum))

            # Backward pass
            net.reset_gradients()
            net.backward(ground_truth=labels)

            # Take optimizer step
            optim.step()
