### How does pruning affect the representational allignment of models?
import os
import torch
import numpy as numpy
from tqdm import tqdm
import pandas as pandas
from my_utils.pruning import prune_model
from pyaml_env import parse_config, BaseConfig
from my_utils.alignment import compute_RSA
from my_utils.imagenet import get_imagenet_dataloader
from my_utils.model_helpers import set_seed, init_model, get_layer_names, get_nice_layer_names, get_layer_activations



def expt_main():

    ## **** Config Args ****
    config = BaseConfig(parse_config('./config.yaml'))
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")
    data_dir = config.imagenet_val
    prune_method = config.prune_method
    alignment_metric = config.alignment_metric
    if alignment_metric == 'rsa':
        compute_alignment = compute_RSA
    alignment_distance = config.alignment_distance
    alignment_correlation = config.alignment_correlation
    
    prune_all = config.prune_all # whether to prune all layers or a single layer
    dataset_size = config.dataset_size

    # setup
    set_seed(0) # for reproducibility
    model = init_model('vgg16', trained=True)
    model.to(device)
    model.eval()
    layer_names = get_layer_names(model)[1:] #ignore the input layer for pruning and evaluation since it doesn't have weights
    nice_layer_names = get_nice_layer_names(model, layer_names)
    data_loader = get_imagenet_dataloader(data_dir, val=True, batch_size=128, num_workers=4, subset_num=dataset_size)
    print(f"Setup complete. Imagenet dataloader (n={len(data_loader.dataset)}) and VGG16 (num_layers={len(nice_layer_names)}) ready.")
    print(f"Pruning method: {prune_method}, Metric: {alignment_metric}, Prune all layers: {prune_all}")

    # results init
    sparsity_levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1] # 0.1 - 10% weights remaining, 1 - all weights
    alignment_results_df = pandas.DataFrame(columns=['layer_name'] + [f'sparsity_{k}' for k in sparsity_levels])

    results_path = config.project_path + '2_self_alignment/results/'
    target_suffix = 'all' if prune_all else 'layerwise'
    os.makedirs(results_path, exist_ok=True)

    for ki, k in enumerate(sparsity_levels):
        model_sparse = None
        if prune_all:
            model = init_model('vgg16', trained=True)
            model_sparse = prune_model(model, 'vgg16', method=prune_method, target='all', sparsity_k=k)
            model_full = init_model('vgg16', trained=True)

        print(f"\nEvaluating sparsity level {k} ({ki+1}/{len(sparsity_levels)}) with {prune_method} pruning.")

        for li, ln in tqdm(enumerate(layer_names), total=len(layer_names), desc="Layerwise Evaluation"):

            if not prune_all:
                model = init_model('vgg16', trained=True)
                model_sparse = prune_model(model, 'vgg16', method=prune_method, target=ln, sparsity_k=k)
                model_full = init_model('vgg16', trained=True) # re-init because pruning is an inplace operation

            nice_ln = nice_layer_names[li]
            full_acts = get_layer_activations(model_full, ln, data_loader, pool_method='global', device_id=config.device_id)
            pruned_acts = get_layer_activations(model_sparse, ln, data_loader, pool_method='global', device_id=config.device_id)

            alignment = compute_alignment(full_acts, pruned_acts, dist_metric=alignment_distance, corr_metric=alignment_correlation)
            
            alignment_results_df.loc[li, 'layer_name'] = nice_ln
            alignment_results_df.loc[li, f'sparsity_{k}'] = alignment.item()

        #save intermediate results
        alignment_results_df.to_csv(os.path.join(results_path, f'vgg_{alignment_metric}_{prune_method}_{target_suffix}.csv'), index=False)

if __name__ == "__main__":
    expt_main()

