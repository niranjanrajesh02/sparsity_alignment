import torch
import random
import numpy as np
from torch import nn
import torch.nn.init as init
from torchvision import models


### --- Model Initialization Helpers --- ###
def set_seed(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False



# Wrapper to include input layer for extracting pixels as activations
class ModelWithInputLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.input_layer = nn.Identity()
        self.model = model
    def forward(self, x):
        x = self.input_layer(x)
        return self.model(x)


def init_model(model_name, seed=0, trained=False):
    assert model_name in [
        'vgg16',
        'resnet18',
        'resnet50',
        'convnext_b',
        'vit_b_16'
    ], f"Model {model_name} not supported."


    model = None
    weights_str = "IMAGENET1K_V1" if trained else None
    if model_name == 'vgg16':
        model = models.vgg16(weights=weights_str)
    elif model_name == 'resnet18':
        model = models.resnet18(weights=weights_str)
    elif model_name == 'resnet50':
        model = models.resnet50(weights=weights_str)
    elif model_name == 'convnext_b':
        model = models.convnext_base(weights=weights_str)   
    elif model_name == 'vit_b_16':
        model = models.vit_b_16(weights=weights_str)

    # add input wrapper
    model = ModelWithInputLayer(model)
    return model    


def get_layer_names(model):
    layer_names = []
    for name, layer in model.named_modules():
        # skip container modules (those that have submodules)
        if len(list(layer.children())) == 0:
            # skip Relu and dropout
            if not isinstance(layer, (torch.nn.ReLU, torch.nn.Dropout)):
                layer_names.append(name)
    return layer_names[:-1]  # exclude the final classifier layer

def get_nice_layer_names(model, layer_names):
    nice_layer_names = []
    named_modules = dict(model.named_modules())
    counters = {}

    for layer in layer_names:
        if layer == 'input_layer':
            nice_layer_names.append('Input')
            continue
        
        module = named_modules.get(layer, None)
        if module is None:
            nice_layer_names.append(layer)
            continue
        
        layer_type = module.__class__.__name__
        counters[layer_type] = counters.get(layer_type, 0) + 1
        nice_layer_names.append(f"{layer_type.replace('2d','')}{counters[layer_type]}")

    return nice_layer_names

### --- Activation Extraction Helpers --- ###

# Extract activations from a specified layer module given image data
def get_layer_activations(model, layer_name, image_data, device_id=0):
    layer = dict(model.named_modules())[layer_name]
    activations = []
    def hook_fn(module, input, output):
          activations.append(output.detach().cpu())
    handle = layer.register_forward_hook(hook_fn)


    model.to(f"cuda:{device_id}")
    with torch.no_grad():
      for images in image_data:
        images = images.to(f"cuda:{device_id}")
        _ = model(images)      
    handle.remove()
  
    acts = torch.cat(activations, dim=0)
    
    # For convolutional layers, apply adaptive average pooling to reduce 
    # spatial dimensions and approx match target_dim when flattened
    target_dim = 4096 # fixing due to fc dimensions in VGG16

    if len(acts.shape) > 2:
      n_channels = acts.shape[1]
      pool_dim = int(np.round(np.sqrt(target_dim/n_channels)))
      apool = nn.AdaptiveAvgPool2d((pool_dim, pool_dim))
      acts = apool(acts)
      acts = acts.flatten(1)
    error_dim = abs(acts.shape[1] - target_dim)
    if error_dim > 1000:
        print(f"Warning: extracted acts dim {acts.shape[1]} differs from target dim {target_dim} by {error_dim} \n Layer: {layer_name}, original shape: {shape_info}")

    # bound acts and remove nans
    acts = acts.nan_to_num_(posinf=1e6, neginf=-1e6, nan=0.0)

    max_val = torch.max(torch.abs(acts))
    # Normalize to keep within [-max_val, max_val]
    if max_val > 1e6 and max_val != 0:
        scale = 1e6 / max_val
        acts = acts * scale
    
    return acts.numpy()

#### --- Misc Helpers --- ###
def count_weights(model):
    ## Counts total and non-zero weights in the model
    total_weights = 0
    nonzero_weights = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            total_weights += param.numel()
            nonzero_weights += torch.sum(param != 0).item()
    return nonzero_weights, total_weights

