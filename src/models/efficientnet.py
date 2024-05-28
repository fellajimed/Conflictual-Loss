from torch import nn

from .mlp import MLPNet
from .models import model_factory
from .classification import ClassificationModuleNet

# logger object
import logging
logger = logging.getLogger('main_all')


class EfficientNet(ClassificationModuleNet):
    def __init__(self,  input_shape, nb_classes, device='cpu', freeze=False,
                 efficientnet_type='b0', pretrained=None, **kwargs_mlp):
        super().__init__(input_shape, nb_classes, device)
        assert efficientnet_type in [
            *list(map(lambda x: f"b{x}", range(7))),
            *list(map(lambda x: f'v2_{x}', 'sml'))]

        # logs
        if (pretrained is not None or
                (bool(pretrained) and isinstance(pretrained, bool))):
            logger.info(f'using pretrained EfficientNet{efficientnet_type}')

        # loading the resnet model
        self.efficientnet = model_factory(f'efficientnet_{efficientnet_type}',
                                          pretrained)

        input_shape_mlp = (self.efficientnet.classifier[1].in_features,)

        self.freeze = freeze
        if freeze:
            logger.info('freezing all layers but the classfier')
            for params in self.efficientnet.parameters():
                params.requires_grad = False

        if kwargs_mlp:
            # changing the classifier with an MLP model
            self.efficientnet.classifier = MLPNet(
                input_shape=input_shape_mlp, nb_classes=nb_classes,
                device=device, **kwargs_mlp)
        else:
            # if the kwargs of the MLP are empty
            # replace the last linear of the classifier
            self.efficientnet.classifier[-1] = nn.Linear(
                input_shape_mlp[0], nb_classes)

        # move model to device
        self.to(device)

    def forward(self, x):
        if self.training:
            if self.freeze:
                self.eval()
                self.efficientnet.classifier.train()
            else:
                self.train()
        return self.efficientnet(x)


class EfficientNet_b0(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b0', **kwargs)


class EfficientNet_b1(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b1', **kwargs)


class EfficientNet_b2(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b2', **kwargs)


class EfficientNet_b3(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b3', **kwargs)


class EfficientNet_b4(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b4', **kwargs)


class EfficientNet_b5(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b5', **kwargs)


class EfficientNet_b6(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b6', **kwargs)


class EfficientNet_b7(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='b7', **kwargs)


class EfficientNetV2_s(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='v2_s', **kwargs)


class EfficientNetV2_m(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='v2_m', **kwargs)


class EfficientNetV2_l(EfficientNet):
    def __init__(self, input_shape, nb_classes, device='cpu', **kwargs):
        super().__init__(input_shape, nb_classes, device,
                         efficientnet_type='v2_l', **kwargs)
