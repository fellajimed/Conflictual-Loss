from torch import nn

from .mlp import MLPNet
from .models import model_factory
from .classification import ClassificationModuleNet

# logger object
import logging
logger = logging.getLogger('main_all')


class ResNet(ClassificationModuleNet):
    def __init__(self,  input_shape, nb_classes, device='cpu', freeze=False,
                 resnet_type=18, pretrained=None, **kwargs_mlp):
        super().__init__(input_shape, nb_classes, device)
        assert resnet_type in [18, 34, 50, 101, 152]

        if resnet_type in [18, 34]:
            input_shape_mlp = (512,)
        else:
            input_shape_mlp = (2048,)

        # logs
        if (pretrained is not None or
                (bool(pretrained) and isinstance(pretrained, bool))):
            logger.info(f'using pretrained ResNet{resnet_type}')

        # loading the resnet model
        self.resnet = model_factory(f'resnet{resnet_type}', pretrained)

        self.freeze = freeze
        if freeze:
            logger.info('freezing all layers but the fc')
            for params in self.resnet.parameters():
                params.requires_grad = False

        if kwargs_mlp:
            # changing the fully connnected layer (fc) with an MLP model
            self.resnet.fc = MLPNet(input_shape=input_shape_mlp,
                                    nb_classes=nb_classes,
                                    device=device, **kwargs_mlp)
        else:
            # if the kwargs of the MLP are empty, replace the fc with a Linear
            self.resnet.fc = nn.Linear(input_shape_mlp[0], nb_classes)

        # move model to device
        self.to(device)

    def forward(self, x):
        if self.training:
            if self.freeze:
                self.eval()
                self.resnet.fc.train()
            else:
                self.train()
        return self.resnet(x)


class ResNet18(ResNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         resnet_type=18, **kwargs)


class ResNet34(ResNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         resnet_type=34, **kwargs)


class ResNet50(ResNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         resnet_type=50, **kwargs)


class ResNet101(ResNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         resnet_type=101, **kwargs)


class ResNet152(ResNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         resnet_type=152, **kwargs)
