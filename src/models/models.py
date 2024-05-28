import torchvision.models as torchmodels

# logger object
import logging
logger = logging.getLogger('main_all')


def model_factory(model_name, value=None):
    """
    model_name: str
    value: bool | str
    """
    from importlib.metadata import version

    if tuple(map(int, version('torchvision').split('.'))) < (0, 13):
        # pretrained expects a bool value
        if isinstance(value, bool):
            pretrained = value
        else:
            pretrained = (value is not None) or isinstance(value, str)
        model = getattr(torchmodels, model_name,
                        torchmodels.resnet18)(pretrained=pretrained)
    else:
        # weights expects a string value
        if isinstance(value, bool):
            value = "IMAGENET1K_V1" if value else None
        elif isinstance(value, str):
            value = value.upper()
        model = getattr(torchmodels, model_name,
                        torchmodels.resnet18)(weights=value)
    return model
