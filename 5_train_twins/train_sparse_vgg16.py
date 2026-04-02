import os
import torch
import numpy as numpy
from tqdm import tqdm
import pandas as pandas
from my_utils.create_network import init_vgg16_custom
from pyaml_env import parse_config, BaseConfig
from my_utils.imagenet import get_imagenet_dataloader
from my_utils.model_helpers import set_seed, init_model, save_model_state



### Adapted from Kapoor2025 https://github.com/NeuroML-Lab/representation-alignment/blob/master/cnns/main_imagenet.py
def train_epoch(model, trainloader, optimizer, criterion, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(trainloader):
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward() # computes gradients for all weights

        # zero out gradients of pruned weights to prevent optimizer from accumulating them
        with torch.no_grad():
            for module in model.modules():
                if hasattr(module, 'weight_mask'):
                    module.weight_orig.grad.mul_(module.weight_mask)

        optimizer.step() # updates all non-pruned weights based on their gradients


        train_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    acc = 100.0 * correct / total
    avg_loss = train_loss / (batch_idx + 1)
    return acc, avg_loss



def test_epoch(model, testloader, criterion, device):
    
    model.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
    
    acc = 100.0 * correct / total
    avg_loss = test_loss / (batch_idx + 1)

    return acc, avg_loss
    

def main():

    ## **** Config Args ****
    config = BaseConfig(parse_config('./config.yaml'))
    device = torch.device(f"cuda:{config.device_id}" if torch.cuda.is_available() else "cpu")
    val_dir = config.imagenet_val
    train_dir = config.imagenet_train

    percent_weights = config.percent_weights
    
    set_seed(0) # for reproducibility
    model = init_vgg16_custom(percent_weights=percent_weights, dropout=False)
    model.to(device)
    print(f"Initialized VGG16 with {percent_weights*100:.1f}% weights remaining and no dropout.")
    base_path = config.project_path + f'5_train_twins/ckpts/k={percent_weights}/'
    os.makedirs(base_path, exist_ok=True)


    ## --- TRAINING PARAMS
    num_epochs = 200
    lr = 0.1
    momentum = 0.9
    weight_decay = 5e-4
    batch_size = 256
    save_freq = 10


    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    criterion = torch.nn.CrossEntropyLoss()
    best_acc = 0.0
    ## ---

    ## -- Restore from checkpoint if exists
    checkpoint_path = base_path + 'restart.pth'
    if os.path.exists(checkpoint_path):
        print(f"Checkpoint found at {checkpoint_path}, restoring model state.")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(checkpoint['model'])
        best_acc = checkpoint.get('val_acc', 0.0)
        print(f"Restored model with validation accuracy: {best_acc:.3f}%")
        optimizer.load_state_dict(checkpoint['optimizer'])
        scheduler.load_state_dict(checkpoint['scheduler'])
        num_epochs -= (checkpoint['epoch'] + 1)  # adjust remaining epochs based on checkpoint epoch
        print(f"Continuing training for {num_epochs} more epochs.")


    # DATA SETUP

    train_dl = get_imagenet_dataloader(train_dir, val=False, batch_size=batch_size, num_workers=4)
    val_dl = get_imagenet_dataloader(val_dir, val=True, batch_size=batch_size, num_workers=4)
    print(f"Setup complete. Imagenet dataloaders (Train_N: {len(train_dl.dataset)}, Val_N: {len(val_dl.dataset)}) and VGG16 ready.")

    #  Train loop
    for epoch_i in tqdm(range(num_epochs)):
        train_acc, train_loss = train_epoch(model, train_dl, optimizer, criterion, device)
        val_acc, val_loss = test_epoch(model, val_dl, criterion, device)
        
        
        # Save checkpoint.
        if val_acc > best_acc:
            state = {
                "model": model.state_dict(),
                "train_acc": train_acc,
                "val_acc": val_acc,
                "epoch": epoch_i,
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(),
            }
            torch.save(state, f"{base_path}/best.pth")
            best_acc = val_acc

        # save weights every save_freq epoch
        if epoch_i == 199:
            print(f"saving weights at epoch {epoch_i}")
            state = {"model": model.state_dict(), "acc": val_acc, "epoch": epoch_i}
            torch.save(state, f"{base_path}/epoch_{epoch_i}.pth")
        if epoch_i % save_freq == 0:
            if epoch_i != 0:
                print(f"saving weights at epoch {epoch_i}")
                state = {"model": model.state_dict(), "acc": val_acc, "epoch": epoch_i}
                torch.save(state, f"{base_path}/epoch_{epoch_i}.pth")
        
        if scheduler:
            scheduler.step()

        # log epoch metrics
        log_path = base_path + '/training_log.txt'
        with open(log_path, 'a') as f:
            f.write(f"epoch={epoch_i} train_acc={train_acc:.3f} train_loss={train_loss:.4f} val_acc={val_acc:.3f} val_loss={val_loss:.4f}\n")


if __name__ == "__main__":
    main()