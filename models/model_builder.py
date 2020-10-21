from torchvision import models
from torch import nn
from .dummy_model import CustomModel


def build_model(model_name):
    if model_name == 'squeezenet':
        model = models.squeezenet1_1(pretrained=True)
        num_of_output_classes = 2
        # change the last conv2d layer
        model.classifier._modules["1"] = nn.Conv2d(512, num_of_output_classes, kernel_size=(1, 1))
        # change the internal num_classes variable rather than redefining the forward pass
        model.num_classes = num_of_output_classes

    elif model_name == 'dummy_model':
        model = CustomModel()
    return model