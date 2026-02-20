### How is weight space pruning affecting the representational space of VGG16?
import os
import torch
import numpy as numpy
from tqdm import tqdm
import pandas as pandas
from my_utils.pruning import prune_model
from pyaml_env import parse_config, BaseConfig
from my_utils.representations_dim import compute_effective_dim, compute_pca_dim
from my_utils.imagenet import get_imagenet_dataloader, eval_model_val
from my_utils.model_helpers import set_seed, init_model, get_layer_names, get_nice_layer_names, get_layer_activations



def extract_sparse_acts():

    ## **** Config Args ****
    config = BaseConfig(parse_config('./config.yaml'))
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")
    data_dir = config.imagenet_val
    prune_method = config.prune_method
    metric = config.metric
    dataset_size = config.dataset_size 


    # setup
    set_seed(0) # for reproducibility
    model = init_model('vgg16', trained=True)
    model.to(device)
    model.eval()
    layer_names = get_layer_names(model)[1:] #ignore the input layer for pruning and evaluation since it doesn't have weights
    nice_layer_names = get_nice_layer_names(model, layer_names)
    data_loader = get_imagenet_dataloader(data_dir, val=True, batch_size=128, num_workers=4, subset_num=dataset_size, shuffle=True)
    print(f"Setup complete. Imagenet dataloader (n={len(data_loader.dataset)}) and VGG16 (num_layers={len(nice_layer_names)}) ready.")

    # results init
    sparsity_levels = [0.1, 0.5, 1] # 0.1 - 10% weights remaining, 1 - all weights
    
    results_path = config.project_path + '1_representation_space/data/test_acts/'
    os.makedirs(results_path, exist_ok=True)
    
    for ki, k in enumerate(sparsity_levels):
        model_sparse = None
        print(f"\nEvaluating sparsity level {k} ({ki+1}/{len(sparsity_levels)}) with {prune_method} pruning.")

        for li, ln in tqdm(enumerate(layer_names), total=len(layer_names), desc="Layerwise Evaluation"):

            model = init_model('vgg16', trained=True)
            model_sparse = prune_model(model, 'vgg16', method=prune_method, target=ln, sparsity_k=k)

            nice_ln = nice_layer_names[li]
            acts = get_layer_activations(model_sparse, ln, data_loader, pool_method='global', device_id=config.device_id)
            numpy.save(os.path.join(results_path, f'vgg_acts_{prune_method}_{ln}_sparsity_{k}.npy'), acts)
            print(f"Saved activations for layer {nice_ln} at sparsity {k}")
    

if __name__ == "__main__":
    extract_sparse_acts()