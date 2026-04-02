import os
import torch
import random
import numpy as np
from torch import nn, seed
import torch.nn.init as init
from torchvision import models
import torch.nn.functional as F 
    

### --- Model Initialization Helpers --- ###
def set_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
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


def init_model(model_name, trained=False, seed=0, state_path=None):
    assert model_name in [
        'vgg16',
        'resnet18',
        'resnet50',
        'convnext_b',
        'vit_b_16'
    ], f"Model {model_name} not supported."


    model = None
    weights_str = "IMAGENET1K_V1" if trained else None
    print(f"Initializing model {model_name} with pretrained={trained}.")
    
    if model_name == 'vgg16':
        model = models.vgg16_bn(weights=weights_str)
    
    # VGG init for untrained
    if trained == False:

        if state_path is not None:
            # get the net property from state dict and load
            state_dict = torch.load(state_path)
            if 'net' in state_dict:
                model.load_state_dict(state_dict['net'])
            else:
                model.load_state_dict(state_dict)
            print(f"Model state loaded from {state_path}")
        else:
            gen = torch.Generator()
            gen.manual_seed(seed)
            for m in model.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu', generator=gen)
                    if m.bias is not None:
                        nn.init.constant_(m.bias, 0)
                elif isinstance(m, nn.Linear):
                    nn.init.normal_(m.weight, mean=0, std=0.01, generator=gen)
                    nn.init.constant_(m.bias, 0)

    model = ModelWithInputLayer(model)

    return model


def load_model(model_name, model_dir, pretrained=False):
    mn = model_name.lower().split('_')[0]
    seed_num = model_name.lower().split('_')[1].replace('seed','')


    assert mn in ["vgg16", "resnet18"], f"Model {model_name} not supported for loading."

    if pretrained: 
        model = init_model(mn, trained=True)
    else:
        model_path = f'{model_dir}/{mn}_trained_seed_{seed_num}.pth'
        model = init_model(mn, trained=False, seed=seed_num, state_path=model_path)

    return model

    




    



def get_layer_names(model, ignore_classifier=True):
    layer_names = []
    # forward pass to identify layers with weights
    for name, module in model.named_modules():
        if name=="input_layer" or isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
            layer_names.append(name)

    if ignore_classifier:
        layer_names = layer_names[:-1]  # exclude the final classifier layer
    
    return layer_names

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
def get_layer_activations(model, layer_name, image_data, pool_method=None, target_dim=4096, device_id=0):
    '''
    collects activations from the specified layer by registering a forward hook.

    Pool Method:
    - if pool_method is None, no pooling is applied, returns full activations as flattened vectors
    - if pool_method is global, applies global pooling to each feature map to get a single value per channel 
        intuition: remove spatial feature info, keep only feature presence info
    - if pool_method is 'adaptive', applies adaptive pooling to get target_dim sized vectors
        intuition: reduce dimensionality to ~target_dim for all layers while preserving some spatial info
    
    '''
    assert pool_method in [None, 'global', 'adaptive'], "pool_method must be one of None, 'global' or 'adaptive'"

    layer = dict(model.named_modules())[layer_name]
    activations = []
    def hook_fn(module, input, output):
          activations.append(output.detach().cpu())
    handle = layer.register_forward_hook(hook_fn)


    model.to(f"cuda:{device_id}")
    model.eval()
    with torch.no_grad():
      for images, _ in image_data:
        images = images.to(f"cuda:{device_id}")
        _ = model(images)      
    handle.remove()
  
    acts = torch.cat(activations, dim=0)
    
    if pool_method == 'global':
        if len(acts.shape) == 4:  
            # conv layer outputs
            if layer_name != 'input_layer':
                acts = F.adaptive_avg_pool2d(acts, (1,1))
                acts = acts.flatten(1)
            
            # input layer outputs (image space)
            else:
                n_channels = acts.shape[1]
                target_dim = 100 # arbitary small to match conv layers
                pool_dim = int(np.round(np.sqrt(target_dim/n_channels)))
                apool = nn.AdaptiveAvgPool2d((pool_dim, pool_dim))
                acts = apool(acts)
                acts = acts.flatten(1)
            

    
    elif pool_method == 'adaptive':
        if len(acts.shape) > 2:
            n_channels = acts.shape[1]
            pool_dim = int(np.round(np.sqrt(target_dim/n_channels)))
            apool = nn.AdaptiveAvgPool2d((pool_dim, pool_dim))
            acts = apool(acts)
            acts = acts.flatten(1)



    elif pool_method is None:
        if len(acts.shape) > 2:
            acts = acts.flatten(1) 
    

    if len(acts.shape) > 2:
      n_channels = acts.shape[1]
      pool_dim = int(np.round(np.sqrt(target_dim/n_channels)))
      apool = nn.AdaptiveAvgPool2d((pool_dim, pool_dim))
      acts = apool(acts)
      acts = acts.flatten(1)
    
    # bound acts and remove nans
    acts = acts.nan_to_num_(posinf=1e6, neginf=-1e6, nan=0.0)
    
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

def save_model_state(model, path):
    torch.save(model.state_dict(), path)