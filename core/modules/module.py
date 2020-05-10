from collections import OrderedDict #TODO: Remove this file if not needed

class Module(object):
    """
    Base class for all neural network modules.
    """

    def __init__(self):
        # Default to train mode for this module.
        self.training = True
        self._parameters = OrderedDict()


    def train(self):
        """
        Sets the model to train mode. This affects certain modules such as batch normalization and dropout.
        Any differences between train and eval mode will be documented in the respective module.

        Returns: None

        """
        self.training = True

    def eval(self):
        """
        Sets the model to eval mode. This affects certain modules such as batch normalization and dropout.
        Any differences between train and eval mode will be documented in the respective module.

        Returns: None

        """
        self.training = False