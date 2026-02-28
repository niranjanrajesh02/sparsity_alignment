
### How is weight space pruning affecting the representational space of VGG16?
import os
import torch
import numpy as numpy
from tqdm import tqdm
import pandas as pandas
from my_utils.pruning import prune_model
from pyaml_env import parse_config, BaseConfig
from my_utils.imagenet import get_imagenet_dataloader
from my_utils.model_helpers import set_seed, init_model, save_model_state
from my_utils.finetune import finetune_classifier, top5_top1_accuracy


def expt_main(prune_method='random'):

    ## **** Config Args ****
    config = BaseConfig(parse_config('./config.yaml'))
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")
    val_dir = config.imagenet_val
    train_dir = config.imagenet_train
    # prune_method = config.prune_method
    reinit_classifier = config.reinit_classifier
    num_val_images = config.dataset_size
    if reinit_classifier:
        num_epochs = 30
    else:
        num_epochs = 20

    # setup
    set_seed(0) # for reproducibility
    model = init_model('vgg16', trained=True)
    model.to(device)
    model.eval()

    train_dl = get_imagenet_dataloader(train_dir, val=False, batch_size=256, num_workers=4)
    val_dl = get_imagenet_dataloader(val_dir, val=True, batch_size=256, num_workers=4, subset_num=num_val_images)
    print(f"Setup complete. Imagenet dataloaders (Train_N: {len(train_dl.dataset)}, Val_N: {len(val_dl.dataset)}) and VGG16 ready.")

    # results init
    sparsity_levels = [0.1, 0.2, 0.5, 0.7,  1] # 0.1 - 10% weights remaining, 1 - all weights
 
    val_acc_results_df = pandas.DataFrame(columns=[f'sparsity_{k}' for k in sparsity_levels]) 

    results_path = config.project_path + '4_retrain/results/'
    reinit_suffix = 'reinit' if reinit_classifier else ''
    os.makedirs(results_path, exist_ok=True)

    for ki, k in enumerate(sparsity_levels):
        model_sparse = None
    
        model = init_model('vgg16', trained=True)
        model_sparse = prune_model(model, 'vgg16', method=prune_method, target='all', sparsity_k=k)

        print(f"\nEvaluating sparsity level {k} ({ki+1}/{len(sparsity_levels)}) with {prune_method} pruning.")
        if k==1:
            print("No pruning applied, evaluating original model.")
            val_acc = top5_top1_accuracy(model, val_dl, device)

        else:
            ft_model, val_acc = finetune_classifier(
                model_sparse,
                train_dl,
                val_dl,
                device,
                reinit_classifier=reinit_classifier,
                num_epochs=num_epochs,
                lr=1e-3,
                weight_decay=1e-4,
                patience=5,
                min_delta=1e-4
            )

        val_acc_results_df.loc[0, f'sparsity_{k}'] = val_acc[0]
        val_acc_results_df.loc[1, f'sparsity_{k}'] = val_acc[1]
        save_model_state(ft_model, os.path.join(results_path, f'vgg16_finetuned_{prune_method}_{k}_{reinit_suffix}.pt'))

        #save intermediate results
        val_acc_results_df.to_csv(os.path.join(results_path, f'vgg_val_acc_{prune_method}_{reinit_suffix}.csv'), index=False)
    

if __name__ == "__main__":
    expt_main(prune_method='random')
    expt_main(prune_method='amp')
    expt_main(prune_method='svd')