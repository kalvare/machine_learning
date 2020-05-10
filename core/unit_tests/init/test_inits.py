"""
This file contains unit tests for initialization methods
"""

import unittest
import torch
import copy
import numpy as np
torch.manual_seed(22)

from core.init.init import xavier_uniform_fully_connected_

class TestInits(unittest.TestCase):

    def setUp(self):
        # Reset random seed
        torch.manual_seed(22)

    def test_xavier_uniform_fc(self):
        w_torch = torch.Tensor(100, 100)
        w = copy.deepcopy(w_torch)

        # Test with Pytorch implementation and compare
        torch.manual_seed(22)
        torch.nn.init.xavier_uniform_(w_torch)
        torch.manual_seed(22)
        xavier_uniform_fully_connected_(w)

        self.assertTrue(np.allclose(w, w_torch))