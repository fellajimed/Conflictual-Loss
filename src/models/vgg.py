from torch import nn

from .mlp import MLPNet
from .models import model_factory
from .classification import ClassificationModuleNet

# logger object
import logging
logger = logging.getLogger('main_all')


class VGG(ClassificationModuleNet):
    def __init__(self,  input_shape, nb_classes, device='cpu', freeze=False,
                 vgg_type=11, pretrained=None, **kwargs_mlp):
        super().__init__(input_shape, nb_classes, device)
        assert str(vgg_type) in ['11', '11_bn', '13', '13_bn',
                                 '16', '16_bn', '19', '19_bn']

        input_shape_mlp = (25088,)

        # logs
        if (pretrained is not None or
                (bool(pretrained) and isinstance(pretrained, bool))):
            logger.info(f'using pretrained VGG{vgg_type}')

        # loading the resnet model
        self.vgg = model_factory(f'vgg{vgg_type}', pretrained)

        self.freeze = freeze
        if freeze:
            logger.info('freezing all layers but the classfier')
            for params in self.vgg.parameters():
                params.requires_grad = False

        if kwargs_mlp:
            # changing the classifier with an MLP model
            self.vgg.classifier = MLPNet(input_shape=input_shape_mlp,
                                         nb_classes=nb_classes,
                                         device=device, **kwargs_mlp)
        else:
            # if the kwargs of the MLP are empty
            # replace the last linear of the classifier
            self.vgg.classifier[-1] = nn.Linear(4096, nb_classes)

        # move model to device
        self.to(device)

    def forward(self, x):
        if self.training:
            if self.freeze:
                self.eval()
                self.vgg.classifier.train()
            else:
                self.train()
        return self.vgg(x)


class VGG11(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='11', **kwargs)


class VGG11_BN(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='11_bn', **kwargs)


class VGG13(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='13', **kwargs)


class VGG13_BN(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='13_bn', **kwargs)


class VGG16(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='16', **kwargs)


class VGG16_BN(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='16_bn', **kwargs)


class VGG19(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='19', **kwargs)


class VGG19_BN(VGG):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         vgg_type='19_bn', **kwargs)
