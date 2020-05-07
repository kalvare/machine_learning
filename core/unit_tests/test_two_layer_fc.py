import unittest
import torch
import numpy as np
import torchvision
import matplotlib.pyplot as plt

from core.unit_tests.grad_check import gradient_check
from core.models.three_layer_fc import SimpleFullyConnected

class TestSimpleFullyConnected(unittest.TestCase):

    def setUp(self):
        random_seed = 1
        torch.backends.cudnn.enabled = False
        torch.manual_seed(random_seed)

    def test_init(self):
        # This just tests that the constructor can run w/o errors
        net = SimpleFullyConnected(in_features=100, hidden_size=50, n_classes=10,
                                   weight_decay=1e-3, weight_init='Kaiming')

    def test_forward(self):
        n_classes = 10
        net = SimpleFullyConnected(in_features=100, hidden_size=50, n_classes=n_classes,
                                   weight_decay=1e-3, weight_init='Kaiming')
        batch_size = 32
        n_features = 100
        mock_data = torch.tensor(data=np.random.randn(batch_size, n_features),
                                 dtype=torch.float,
                                 requires_grad=False)
        mock_probs = net.forward(mock_data)

        # Assert correct shape
        self.assertTrue(mock_probs.shape == (batch_size, n_classes))

        # Assert that the outputs are probability distributions
        for batch in range(mock_probs.shape[0]):
            sum = np.sum(mock_probs[batch, :].numpy())
            self.assertTrue(np.isclose(sum, 1.0))
        print()

    def test_one_hot(self):
        n_classes = 10
        net = SimpleFullyConnected(in_features=100, hidden_size=50, n_classes=n_classes,
                                   weight_decay=1e-3, weight_init='Kaiming')
        labels = torch.tensor([0, 3, 7, 5])
        expected = torch.tensor([[1., 0., 0., 0., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 1., 0., 0., 0., 0., 0., 0.],
                                 [0., 0., 0., 0., 0., 0., 0., 1., 0., 0.],
                                 [0., 0., 0., 0., 0., 1., 0., 0., 0., 0.]])

        one_hot = net._one_hot(labels)
        self.assertTrue((one_hot == expected).all())

    def test_back_prop(self):
        n_classes = 10
        batch_size = 4
        n_features = 100

        net = SimpleFullyConnected(in_features=100, hidden_size=50, n_classes=n_classes,
                                   weight_decay=1e-3, weight_init='Kaiming')

        labels = torch.tensor([0, 3, 7, 5])
        mock_data = torch.tensor(data=np.random.randn(batch_size, n_features),
                                 dtype=torch.float,
                                 requires_grad=False)
        mock_probs = net.forward(mock_data)
        net.backward(labels)

    def test_backward(self):
        # Test backward function with gradient checking
        n_classes = 10
        n_features = 28 * 28

        net = SimpleFullyConnected(in_features=n_features, hidden_size=50, n_classes=n_classes,
                                   weight_decay=0, weight_init='Kaiming', dtype=torch.double)

        batch_size = 128
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST('data', train=True, download=True,
                                       transform=torchvision.transforms.Compose([
                                           torchvision.transforms.ToTensor(),
                                           torchvision.transforms.Normalize(
                                               (0.1307,), (0.3081,))
                                       ])),
            batch_size=batch_size, shuffle=False, drop_last=True)
        batch_generator = iter(train_loader)

        x, y = next(batch_generator)
        x = x.view(batch_size, -1).double()

        ### Run gradient check for a handful of parameters in each weight matrix/vector in the model ###
        # Start with W_1
        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.W_1[5, 12],
                                               backward_grad_idx=(0, 5, 12),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.W_1[17, 46],
                                               backward_grad_idx=(0, 17, 46),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

        # Now with W_2
        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.W_2[5, 12],
                                               backward_grad_idx=(2, 5, 12),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.W_2[9, 46],
                                               backward_grad_idx=(2, 9, 46),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

        # Now with b_1
        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.b_1[5],
                                               backward_grad_idx=(1, 5),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.b_1[9],
                                               backward_grad_idx=(1, 9),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

        # Now with b_2
        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.b_2[3],
                                               backward_grad_idx=(3, 3),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

        abs_err, relative_err = gradient_check(x=x, y=y, model=net, parameter=net.b_2[6],
                                               backward_grad_idx=(3, 6),
                                               cost_fn=torch.nn.functional.cross_entropy,
                                               kwargs={'reduction': 'mean'})
        self.assertTrue(np.isclose(abs_err, 0))
        print('absolute error: %.10f\nrelative error: %.10f' % (abs_err, relative_err))

    def test_minibatch_gradient_decent(self):

        n_classes = 10
        n_features = 28 * 28

        net = SimpleFullyConnected(in_features=n_features, hidden_size=150, n_classes=n_classes,
                                   weight_decay=0, weight_init='Kaiming')

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

        losses = []
        for i in range(1000):
            print(i)
            try:
                imgs, labels = next(batch_generator)
            except StopIteration:
                break
            imgs = imgs.view(batch_size, -1)

            # Forward pass
            unnormalized_log_probs = net.forward(imgs, return_type='unnormalized_log_probs')

            loss = torch.nn.functional.cross_entropy(input=unnormalized_log_probs, target=labels)
            losses.append(loss)

            # Backward pass
            net.reset_gradients()
            net.backward(ground_truth=labels)

            net.W_1 -= 0.001 * net.gradient_W_1
            net.W_2 -= 0.001 * net.gradient_W_2
            net.b_1 -= 0.001 * net.gradient_b_1
            net.b_2 -= 0.001 * net.gradient_b_2


        plt.plot(losses)
        plt.show()






