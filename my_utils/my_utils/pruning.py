import torch
import random
import numpy as np
from torch import nn
from torchvision import models



### --- Helper Functions --- ###
def verify_sparsity(W_s, sparsity_k, name):
    # Verify sparsity to be within tolerance (5%)
    actual_nonzero = torch.sum(W_s != 0).item()
    actual_sparsity = actual_nonzero / W_s.numel() # fraction of weights kept
    assert abs(actual_sparsity - sparsity_k) < 0.05, f"Sparsity check failed for layer {name}: expected {sparsity_k}, got {actual_sparsity}"


### --- Pruning Methods --- ###
def svd_prune_matrix(module, sparsity_k, return_count=False, MHA_flag=False):
    # "Prune" the weight matrix using low-rank approximation via SVD
    # sparsity_k is the fraction of effective weights needed to be retained
    # effective weights are the number of parameters in the low-rank representation 

    W = module.weight if not MHA_flag else module.in_proj_weight
    m,n = W.shape
    max_rank = min([m,n])
    rank_k = max(1, round((sparsity_k * m * n) / (m + n + 1))) # rank needed to retain k fraction of weights
 
    U, S, Vh = torch.linalg.svd(W, full_matrices=False)
    U_k = U[:, :rank_k]
    S_k = torch.diag(S[:rank_k])
    Vh_k = Vh[:rank_k, :]
    W_lowrank = torch.matmul(U_k, torch.matmul(S_k, Vh_k))

    if return_count:
        eff_param_count = U_k.numel() + rank_k + Vh_k.numel()
        return W_lowrank, eff_param_count

    else:
        return W_lowrank


def prune_matrix(module, sparsity_k, prune_method='random', MHA_flag=False):
    assert prune_method in ['random', 'amp']

    W = module.weight if not MHA_flag else module.in_proj_weight
    with torch.no_grad():
        W_flat = W.view(-1)
        num_weights_to_keep = round(sparsity_k * W_flat.numel()) # number of weights to keep so that k fraction is retained
        

        if prune_method == 'svd':
            W_sparse = svd_prune_matrix(module, sparsity_k, MHA_flag=MHA_flag)

        else:
            if prune_method == 'random':
                indices = torch.randperm(W_flat.numel())[:num_weights_to_keep]
            elif prune_method == 'amp':
                _, indices = torch.topk(torch.abs(W_flat), num_weights_to_keep)

            mask = torch.zeros_like(W_flat)
            mask[indices] = 1
            W_sparse = (W_flat * mask).view_as(W)

        module.weight.copy_(W_sparse) if not MHA_flag else module.in_proj_weight.copy_(W_sparse)
    return W_sparse


### --- Main Pruning Function --- ###
def prune_model(model, arch, method='random', sparsity_k=0.5):
    """
    Prune the weights of the given model by zeroing out a fraction of the weights.
        For conv layers, pruning is achieved randomly on weights across all channels.
        For linear layers, pruning is achieved randomly on the weight matrix.
    Args:
        model: The neural network model to prune.
        arch: Architecture name
        method: Pruning method ('random', 'amp' and 'svd' supported).
        sparsity_k: Fraction of weights to keep (between 0 and 1).
    Returns:
        The pruned model.

    """

    assert 0.0 <= sparsity_k <= 1.0, "sparsity_k must be between 0 and 1."
    assert arch in ["vgg16", "resnet18", "vit_b_16", "convnext_b"]

    layer_names = []
    # forward pass to identify layers with weights
    for name, module in model.named_modules():
        if isinstance(module, (nn.Conv2d, nn.Linear, nn.MultiheadAttention)):
            layer_names.append(name)


    layer_names = layer_names[:-1]  # exclude the final classification layer
    layers_to_sparsify = layer_names #TODO: Add options for target_layers
    print(f'Pruning {len(layers_to_sparsify)} layers randomly with sparsity k={sparsity_k}')

    if sparsity_k == 1.0:
        print("Sparsity k=1.0, no pruning applied.")
        return model
    
    
    modules = dict(model.named_modules())
    
    for name in layers_to_sparsify:
        module = modules[name]
        W_s = None

        #**CONV and LINEAR**
        if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear):
            W_s = prune_matrix(module, sparsity_k, prune_method=method)

        ##**MULTIHEAD ATTENTION** (Attn_in layers in ViT, Attn_out layers handled as Linear)
        elif isinstance(module, nn.MultiheadAttention):
            W_s = prune_matrix(module, sparsity_k, prune_method=method, MHA_flag=True)
            
        # Verify sparsity to be within tolerance (5%)
        if W_s is not None:
            verify_sparsity(W_s, sparsity_k, name)

    return model