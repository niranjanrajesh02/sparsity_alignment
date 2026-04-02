
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
from my_utils.finetune import train_vgg_model, top5_top1_accuracy


def expt_main():

    ## **** Config Args ****
    config = BaseConfig(parse_config('./config.yaml'))
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")
    val_dir = config.imagenet_val
    train_dir = config.imagenet_train
    seed = config.seed
    num_epochs = 100
    # setup
    set_seed(seed) # for reproducibility
    model = init_model('vgg16', trained=False, seed=seed)
    model.to(device)
    model.eval()

    train_dl = get_imagenet_dataloader(train_dir, val=False, batch_size=256, num_workers=4)
    val_dl = get_imagenet_dataloader(val_dir, val=True, batch_size=256, num_workers=4)
    print(f"Setup complete. Imagenet dataloaders (Train_N: {len(train_dl.dataset)}, Val_N: {len(val_dl.dataset)}) and VGG16 ready.")

    # results init
    results_path = config.project_path + '/saved_models/'
    os.makedirs(results_path, exist_ok=True)
    trained_model, val_acc = train_vgg_model(
        model,
        train_dl,
        val_dl,
        device,
        num_epochs=num_epochs
    )

    # Save the finetuned model and results
    save_model_state(trained_model, results_path + f'vgg16_trained_seed_{seed}.pt')

       
    

if __name__ == "__main__":
    expt_main()