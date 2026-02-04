import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset



def get_imagenet_dataloaders(train_dir, val_dir,batch_size=128, num_workers=4):
  train_tf = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                          std=[0.229,0.224,0.225]),
  ])
  val_tf = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                          std=[0.229,0.224,0.225]),
  ])
  
  trainsub_loader = DataLoader(
      datasets.ImageFolder(train_dir, train_tf),
      batch_size=batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True
  )


  # subset 1000 imgs for quick eval
  random_idx = torch.randperm(len(trainsub_loader.dataset))[:1000]

  train_loader = torch.utils.data.DataLoader(
      Subset(trainsub_loader.dataset, random_idx),
      batch_size=trainsub_loader.batch_size,
      shuffle=False,
      num_workers=trainsub_loader.num_workers,
  )

  val_loader = DataLoader(
      datasets.ImageFolder(val_dir, val_tf),
      batch_size=batch_size, shuffle=False,
      num_workers=num_workers, pin_memory=True
  )


  return train_loader, val_loader



def eval_model_val(model, val_dl, device_id=0):
  device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  correct_top1 = 0
  correct_top5 = 0
  total = 0

  with torch.no_grad():
    for images, labels in tqdm(val_dl, desc="Evaluating Model on Val Set"):
      images = images.to(device)
      labels = labels.to(device)

      logits = model(images)
      _, pred_top5 = logits.topk(5, 1, True, True)
      total += labels.size(0)
      correct = pred_top5.eq(labels.view(-1,1).expand_as(pred_top5))
      correct_top1 += correct[:, :1].sum().item()
      correct_top5 += correct.sum().item()

  val_acc_top1 = correct_top1 / total
  val_acc_top5 = correct_top5 / total

  return val_acc_top1, val_acc_top5




