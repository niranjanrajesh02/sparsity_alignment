### How does pruning affect the representational allignment of models?
import os
import torch
import numpy as numpy
from torchgen import model
from tqdm import tqdm
import pandas as pandas
from my_utils.pruning import prune_model
from pyaml_env import parse_config, BaseConfig
from my_utils.imagenet import get_imagenet_dataloader
from my_utils.model_helpers import set_seed, init_model, load_model, get_layer_names, get_nice_layer_names, get_layer_activations
from safetensors.torch import save_file, load_file





def expt_main():

    ## **** Config Args ****
    config = BaseConfig(parse_config('./config.yaml'))
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")
    data_dir = config.imagenet_val
    prune_method = config.prune_method    
    dataset_size = config.dataset_size
    seed = config.seed
    
    model_name = "vgg16_seed1"
    model_dir = config.project_path + '/saved_models/'
    prune_method_s = prune_method

    if prune_method == "random":
        prune_method_s = f"{prune_method}_"+str(seed)
    
    # setup
    set_seed(seed) # for reproducibility
    model = load_model(model_name, model_dir)
    model.to(device)
    model.eval()
    layer_names = get_layer_names(model)[1:] #ignore the input layer for pruning and evaluation since it doesn't have weights
    nice_layer_names = get_nice_layer_names(model, layer_names)
    data_loader = get_imagenet_dataloader(data_dir, val=True, batch_size=256, num_workers=4, subset_num=dataset_size)
    print(f"Setup complete. Imagenet dataloader (n={len(data_loader.dataset)}) and VGG16 (num_layers={len(nice_layer_names)}) ready.")

    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 0.1 - 10% weights remaining, 1 - all weights

    save_path = config.project_path + 'activations_cache/'
    target_suffix = 'all' 
    os.makedirs(save_path, exist_ok=True)
    ln = layer_names[-1]

    acts_cache = {}
    for ki, k in enumerate(sparsity_levels):
        model_sparse = None
        model = load_model(model_name, model_dir)
        model_sparse = prune_model(model, 'vgg16', method=prune_method, target='all', sparsity_k=k)
        acts_cache[k] = get_layer_activations(model_sparse, ln, data_loader, pool_method='global', device_id=config.device_id)
        print(f"Extracted activations for sparsity level {k} ({ki+1}/{len(sparsity_levels)}).")

    # store activations arrays as a safetensors file for later retrieval
    cache = {str(k): torch.from_numpy(arr) for k, arr in acts_cache.items()}
    save_file(cache, f"{save_path}{model_name}_acts_{prune_method_s}_{target_suffix}.safetensors")



if __name__ == "__main__":
    expt_main()