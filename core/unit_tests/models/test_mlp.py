import unittest
import torch
import torchvision
import matplotlib.pyplot as plt

from core.models.mlp import MLP
from core.activations.activations import relu
from core.init.init import xavier_uniform_fully_connected_
from core.optimizers.SGD import SGD

class TestMLP(unittest.TestCase):

    def setUp(self):
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)

    def test_minibatch_gradient_decent(self):

        n_classes = 10
        n_features = 28 * 28

        net = MLP(in_features=n_features, hidden_sizes=[150, 150], hidden_activation=relu, n_out=n_classes,
                  init_w=xavier_uniform_fully_connected_, init_b=torch.nn.init.zeros_)

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

        optimizer = SGD(params=net.parameters(), lr=0.001, weight_decay=0.0)

        losses = []
        for i in range(1000):
            print(i)
            try:
                imgs, labels = next(batch_generator)
            except StopIteration:
                break
            imgs = imgs.view(batch_size, -1)

            # Forward pass
            unnormalized_log_probs = net(imgs)

            loss = torch.nn.functional.cross_entropy(input=unnormalized_log_probs, target=labels)
            losses.append(loss)

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        plt.plot(losses)
        plt.show()