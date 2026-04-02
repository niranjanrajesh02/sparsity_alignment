import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import pandas
import os
import copy

from my_utils.model_helpers import get_layer_names



class EarlyStopping:

    def __init__(self, patience=5, min_delta=1e-4):
        self.patience = patience
        self.min_delta = min_delta
        self.best_val_acc = -float('inf')
        self.best_state = None
        self.counter = 0
        self.should_stop = False

    def step(self, val_acc, model):
        if val_acc > self.best_val_acc + self.min_delta:
            self.best_val_acc = val_acc
            self.best_state = copy.deepcopy(model.state_dict())
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.should_stop = True

    def restore_best(self, model):
        model.load_state_dict(self.best_state)
        return model


def freeze_params(model, except_name=None, reinit=False):
    for name, param in model.named_parameters():
        if except_name is None:
            param.requires_grad = False
        else:
            if except_name in name:
                param.requires_grad = True
                print(f"Unfreezing {name} for finetuning.")
                if reinit:
                    print(f"Reinitializing {name} for finetuning.")
                    if param.dim() > 1:  
                        nn.init.normal_(param, mean=0, std=0.01)
                    else:
                        nn.init.constant_(param, 0)
            else:
                param.requires_grad = False
    return model


def train_epoch(model, loader, optim, criterion, device):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    for inputs, labels in tqdm(loader, desc="  Training", leave=False):
        inputs, labels = inputs.to(device), labels.to(device)
        optim.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optim.step()

        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        correct += predicted.eq(labels).sum().item()
        total += labels.size(0)

    return running_loss / total, correct / total

def evaluate_epoch(model, loader, criterion, device):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Evaluating", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            correct += predicted.eq(labels).sum().item()
            total += labels.size(0)

    return running_loss / total, correct / total

def top5_top1_accuracy(model, loader, device):
    model.eval()
    correct_top1, correct_top5, total = 0, 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="  Evaluating Final Accuracy", leave=False):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.topk(5, dim=1)

            correct_top1 += predicted[:, 0].eq(labels).sum().item()
            correct_top5 += predicted.eq(labels.view(-1, 1)).sum().item()
            total += labels.size(0)

    top1_acc = correct_top1 / total
    top5_acc = correct_top5 / total
    return top1_acc, top5_acc


def finetune_classifier(
    model,
    train_loader,
    val_loader,
    device,
    reinit_classifier=False,
    num_epochs=5,
    lr=1e-3,
    weight_decay=1e-4,
    patience=5,
    min_delta=1e-4,
    momentum=0.9
):
    ft_model = copy.deepcopy(model)  # don't mutate the original
    classifier_name = get_layer_names(ft_model, ignore_classifier=False)[-1]  
    ft_model = freeze_params(ft_model, except_name=classifier_name, reinit=reinit_classifier)
    ft_model.to(device)

    criterion = nn.CrossEntropyLoss()

    trainable_params = filter(lambda p: p.requires_grad, ft_model.parameters())
    optimizer = optim.SGD(trainable_params, lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs  
    )


    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(ft_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_epoch(ft_model, val_loader, criterion, device)
        scheduler.step()

        print(
            f"  Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}",
            f"[ES counter: {early_stopping.counter}/{patience}]"
        )

        early_stopping.step(val_acc, ft_model)
        if early_stopping.should_stop:
            print("Early stopping triggered. Restoring best model state.")
            break

     # restore best checkpoint
    ft_model = early_stopping.restore_best(ft_model)

    # get top1 and top5 accuracy
    top1_acc, top5_acc = top5_top1_accuracy(ft_model, val_loader, device)

    print(f"Final Top-1 Accuracy: {top1_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}")

    return ft_model, (top1_acc, top5_acc)

def train_vgg_model(
    model,
    train_loader,
    val_loader,
    device,
    num_epochs=100,
    lr=1e-2,
    weight_decay=5e-4,
    momentum=0.9,
    patience=5,
    min_delta=1e-4
):
    t_model = copy.deepcopy(model)  # don't mutate the original
    t_model.to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.SGD(t_model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=200)

    early_stopping = EarlyStopping(patience=patience, min_delta=min_delta)

    for epoch in range(num_epochs):
        train_loss, train_acc = train_epoch(t_model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate_epoch(t_model, val_loader, criterion, device)
        scheduler.step()


        print(
            f"  Epoch [{epoch+1}/{num_epochs}] "
            f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}",
            f"[ES counter: {early_stopping.counter}/{patience}]"
        )

        early_stopping.step(val_acc, t_model)
        if early_stopping.should_stop:
            print("Early stopping triggered. Restoring best model state.")
            break

     # restore best checkpoint
    t_model = early_stopping.restore_best(t_model)

    # get top1 and top5 accuracy
    top1_acc, top5_acc = top5_top1_accuracy(t_model, val_loader, device)

    print(f"Final Top-1 Accuracy: {top1_acc:.4f}, Top-5 Accuracy: {top5_acc:.4f}")

    return t_model, (top1_acc, top5_acc)