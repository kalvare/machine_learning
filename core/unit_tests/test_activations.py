import unittest
import torch
import numpy as np

from core.activations.activations import relu_grad

class TestActivations(unittest.TestCase):

    def test_relu_gradient(self):
        x = torch.zeros(4, 100)
        x[2, 62] = 1
        x[3, 79] = 15
        x[1, 43] = 0
        x[0, 29] = -5

        grad = relu_grad(x)
        self.assertTrue(grad[2, 62] == 1)
        self.assertTrue(grad[3, 79] == 1)
        self.assertTrue(grad[1, 43] == 0)
        self.assertTrue(grad[0, 29] == 0)
