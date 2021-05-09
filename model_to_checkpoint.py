import torch
from torch import nn

from torchvision import models

import os
import argparse
import sys

from collections import OrderedDict

parser = argparse.ArgumentParser(description='PyTorch CelebaModel to checkpoint converter')

parser.add_argument('--checkpoint', required=True)
parser.add_argument('--model', required=True)
parser.add_argument('--epoch', required=True, type=int)

args = parser.parse_args()

class CelebaModel(nn.Module):
    def __init__(self):
        super(CelebaModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(2048, 40)

    def forward(self, x):
        return self.resnet50(x)

class CelebaModel18(nn.Module):
    def __init__(self):
        super(CelebaModel18, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.fc = nn.Linear(512, 40)

    def forward(self, x):
        return self.resnet18(x)

model = torch.load(args.model)

state_dict = model.state_dict()
model_type = list(state_dict.keys())[0].split('.')[0]

assert model_type in ['resnet50', 'resnet18'], f'unrecognized model type: {model_type}'

def chop_module(x):
    t = model_type + '.'
    assert x.startswith(t)
    return x[len(t):]

state = {
    'net': OrderedDict([(chop_module(key), value)
                        for key, value in state_dict.items()]),
    # 'acc': acc,
    'epoch': args.epoch,
    'model': model_type,
    'normalize': False,
}

checkpoint_dir = 'checkpoints'
if not os.path.isdir(checkpoint_dir):
    os.mkdir(checkpoint_dir)

checkpoint_file = os.path.join(checkpoint_dir, args.checkpoint)
torch.save(state, checkpoint_file.format(args.epoch))
