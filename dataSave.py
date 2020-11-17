import torch
import torchvision
import torch.nn as nn
import math
import torch.nn.functional as F
import numpy as np

num_epochs = 10
batch_size_train = 100
batch_size_test = 100
learning_rate = 0.001
momentum = 0.5
log_interval = 500


train_loader = torch.utils.data.DataLoader(
  torchvision.datasets.EMNIST(root='./trainData/', split='bymerge', train=True, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.RandomPerspective(), 
                               torchvision.transforms.RandomRotation(47), 
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_train, shuffle=True)


test_loader = torch.utils.data.DataLoader(
  torchvision.datasets.EMNIST(root='./testData/', split='bymerge', train=False, download=True,
                             transform=torchvision.transforms.Compose([
                               torchvision.transforms.ToTensor(),
                               torchvision.transforms.Normalize(
                                 (0.1307,), (0.3081,))
                             ])),
  batch_size=batch_size_test, shuffle=True)
