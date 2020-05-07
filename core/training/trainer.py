"""
This file contains code for a model training class.
"""

import numpy as np
import torch

from core.loss_functions.cross_entropy import penalized_cross_entropy


class ClassificationTrainer(object):
    """
    Class for training a neural network for classification with penalized cross-entropy loss.
    """
    def __init__(self, data_loader, network, optimizer, len_training, lr_scheduler=None):
        """
        Constructor.
        Args:
            data_loader: torch.DataLoader,
                 data loader to generate batches of data for training.
            network: network,
                 neural network to train.
            optimizer: optimizer,
                 optimizer to use.
            lr_scheduler: torch.optim.lr_scheduler, #TODO: Change to learning rate decay
                 learning rate scheduler
            len_training: int,
                 the number of samples in the training data set

        """
        # TODO: Implement lr scheduler

        self.data_loader = data_loader
        self.network = network
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.epoch = 1
        self.len_training = len_training

        self.avg_loss_per_epoch = []
        self.training_accuracies = []
        self.batch_sizes = []
        self.validation_avg_losses = []
        self.validation_accuracies = []


    def train(self, num_epochs, device, validation_data_loader, len_validation, verbose=True):
        """
        Trains the neural network to minimize the penalized cross-entropy on the training data.
        Args:
            num_epochs: int,
                 number of epochs to train for.
            device: torch.device,
                 device to train on.
            validation_data_loader: torch.DataLoader,
                 data loader to generate batches of validation data.
            len_validation: int,
                 the number of samples in the validation data set.
            verbose: bool,
                 if true, print per epoch progress.

        Returns:
            self.avg_loss_per_epoch: list,
                 current list of average training losses per epoch
            self.training_accuracies: list,
                 current list of training accuracy for each epoch
            self.validation_avg_losses: list,
                 current list of average validation losses per epoch
            self.validation_accuracies: list,
                 current list of validation accuracy for each epoch

        """

        # batch_size = self.data_loader.batch_size
        batch_generator = iter(self.data_loader)

        loss_per_step = []
        batch_sizes = []
        y_true = []
        y_pred = []
        while self.epoch <= num_epochs:
            try:
                imgs, labels = next(batch_generator)
            except StopIteration:
                # Calculate training loss and accuracy
                training_loss = np.average(loss_per_step,
                                           weights=np.array(batch_sizes) / self.len_training)
                self.avg_loss_per_epoch.append(training_loss)
                training_accuracy = np.sum((np.array(y_pred) == np.array(y_true))) / len(y_pred)
                self.training_accuracies.append(training_accuracy)

                # Calculate validation loss and accuracy
                validation_avg_loss, validation_accuracy = self.validate_model(validation_data_loader, len_validation)
                self.validation_avg_losses.append(validation_avg_loss)
                self.validation_accuracies.append(validation_accuracy)

                # Report progress
                if verbose:
                    print('epoch (%05d) -- training loss: (%.5f) -- training accuracy: (%.5f) -- '
                          'validation loss (%.5f) -- '
                          'validation accuracy: (%.5f)' %
                          (self.epoch, training_loss, training_accuracy, validation_avg_loss, validation_accuracy))

                # Set up for next epoch
                batch_generator = iter(self.data_loader)
                imgs, labels = next(batch_generator)
                self.epoch += 1
                loss_per_step = []
                batch_sizes = []
                y_true = []
                y_pred = []

            # Accumulate ground truth predictions over epoch to compute training accuracy
            y_true.extend(labels.cpu().tolist())

            batch_size = imgs.shape[0]
            batch_sizes.append(batch_size)
            imgs = imgs.view(batch_size, -1)
            imgs = imgs.to(device)
            labels = labels.to(device)

            # Forward pass
            log_probs = self.network.forward(imgs, return_type='log_probs')

            loss = penalized_cross_entropy(log_probs=log_probs, classes=labels, model_params=self.network.get_params(),
                                           reduce='mean', weight_decay=self.optimizer.weight_decay)
            loss_per_step.append(loss.item())

            # Make training predictions and accumulate to compute epoch training accuracy
            y_pred_batch = torch.argmax(log_probs, dim=1, keepdim=False)
            y_pred.extend(y_pred_batch.cpu().tolist())

            # Backward pass
            self.network.reset_gradients()
            self.network.backward(ground_truth=labels)

            # Take optimizer step
            self.optimizer.step()

        return self.avg_loss_per_epoch, self.training_accuracies, self.validation_avg_losses, self.validation_accuracies


    def validate_model(self, validation_data_loader, len_validation):
        """
        Assesses the average loss and accuracy on the validation data.
        Args:
            validation_data_loader:
                 data loader to generate batches of validation data.
            len_validation: int,
                 the number of samples in the validation data set.
        Returns:
            validation_avg_loss: float,
                 average loss on the validation data.
            validation_accuracy: float,
                 accuracy on the validation data.

        """

        batch_generator = iter(validation_data_loader)

        loss_per_step = []
        batch_sizes = []
        y_true = []
        y_pred = []
        for x, y in iter(batch_generator): #TODO: Test this
            # Keep track of the actual size of each batch for weighted averaging
            batch_size = x.shape[0]
            batch_sizes.append(batch_size)
            x = x.view(batch_size, -1)

            # Forward pass
            log_probs = self.network.forward(x, return_type='log_probs')

            # Calculate loss
            loss = penalized_cross_entropy(log_probs=log_probs, classes=y, model_params=self.network.get_params(),
                                           reduce='mean', weight_decay=self.optimizer.weight_decay)
            loss_per_step.append(loss)

            y_pred_batch = torch.argmax(log_probs, dim=1, keepdim=False)
            y_pred.extend(y_pred_batch.cpu().tolist())
            y_true.extend(y.cpu().tolist())

        # Calculate average loss over entire data set
        validation_avg_loss = np.average(loss_per_step, weights=np.array(batch_sizes) / len_validation)

        # Calculate the accuracy over the entire data set
        validation_accuracy = np.sum((np.array(y_pred) == np.array(y_true))) / len(y_pred)

        return validation_avg_loss, validation_accuracy