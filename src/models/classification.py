from torch import nn


class ClassificationModuleNet(nn.Module):
    """
    Base class for Classification networks
    """

    def __init__(self, input_shape, nb_classes, device):
        super().__init__()

        # device
        self.device = device
        # shape of inputs
        self.input_shape = input_shape
        # number of classes
        self.output_dim = nb_classes

    def forward(self, x, *args):
        raise NotImplementedError(f"Module [{type(self).__name__}] is",
                                  " missing the required \"forward\" function")
