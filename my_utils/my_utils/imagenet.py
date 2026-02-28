import torch
import torch.nn as nn
from tqdm import tqdm
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torch.utils.data import Subset
import numpy as np


def get_imagenet_dataloader(data_dir, val=True,batch_size=128, num_workers=4, subset_num=None):
  tf=None
  if val:
    tf = transforms.Compose([
      transforms.Resize(256),
      transforms.CenterCrop(224),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                          std=[0.229,0.224,0.225]),
  ])
  else:
    tf = transforms.Compose([
      transforms.RandomResizedCrop(224),
      transforms.RandomHorizontalFlip(),
      transforms.ToTensor(),
      transforms.Normalize(mean=[0.485,0.456,0.406],
                          std=[0.229,0.224,0.225]),
  ])
  
  dataset = datasets.ImageFolder(data_dir, tf)
    
  # Apply subset if specified
  if subset_num is not None:
      # indices = torch.randperm(len(dataset))[:subset_num] # random subset (not class balanced)
      targets = np.array(dataset.targets)          
      classes = np.unique(targets)
      samples_per_class = subset_num // len(classes)
      if samples_per_class > 0:
        indices = np.concatenate([
            np.random.choice(np.where(targets == c)[0], samples_per_class, replace=False)
            for c in classes
        ]).tolist()
      else:
        indices = torch.randperm(len(dataset))[:subset_num].tolist() # if subset_num is smaller than number of classes, just take random subset (not class balanced)

      dataset = Subset(dataset, indices)
  
  # Create DataLoader once
  dl = DataLoader(
      dataset,
      batch_size=batch_size,
      shuffle=False,
      num_workers=num_workers,
      pin_memory=True
  )
  
  return dl

def extract_labels(dataloader):
    labels = []
    for _, y in dataloader:
        labels.append(y)
    return torch.cat(labels).numpy()

def eval_model_val(model, val_dl, device_id=0):
  device = torch.device(f'cuda:{device_id}' if torch.cuda.is_available() else 'cpu')
  model.to(device)
  model.eval()

  correct_top1 = 0
  correct_top5 = 0
  total = 0

  with torch.no_grad():
    for images, labels in val_dl:
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




