import torch
import os
import torch.nn as nn
from torchvision import models
import torch.nn.utils.prune as prune



# create custom vgg16 model with only k% weights in each layer
def init_vgg16_custom(percent_weights=1, dropout=False):
    model = models.vgg16(weights=None)

    if not dropout:
        classifier_layers = []
        for layer in model.classifier:
            if not isinstance(layer, torch.nn.Dropout):
                classifier_layers.append(layer)
        model.classifier = torch.nn.Sequential(*classifier_layers)

    if percent_weights < 1.0:
        sparsity = 1.0 - percent_weights

        prunable = [
            m for m in model.modules()
            if isinstance(m, (torch.nn.Conv2d, torch.nn.Linear))
        ]
        prunable = prunable[:-1]  # exclude final classification Linear

        for module in prunable:
            prune.random_unstructured(module, name='weight', amount=sparsity)

    return model