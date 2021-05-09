import os
import argparse
import sys

import tqdm

import torch
import torch.backends.cudnn as cudnn
from torch import functional as F
from torch import nn
import torchvision
from torchvision import datasets, transforms, models
from torch.utils.tensorboard import SummaryWriter

import resnet

from smoothing import (AverageCounter, MovingAverageCounter,
                       mask_batch, yes_or_no_p)


parser = argparse.ArgumentParser(description='PyTorch Celeba uncertified accuracy calculation')
parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--checkpoint', help='checkpoint', required=True)
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--test', action='store_true',
                    help='Use test set instead of validation')
parser.add_argument('--normalize', action='store_true',
                    help='Normalize set before feeding it to the model')
parser.add_argument('--rewrite-accuracy', action='store_true',
                    help='Normalize set before feeding it to the model')

args = parser.parse_args()

CUDA = torch.cuda.is_available()

if CUDA:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

img_mean = [0.5063, 0.4258, 0.3832]
img_std = [0.2660, 0.2452, 0.2414]

if args.normalize:
    normalize = transforms.Normalize(mean=img_mean, std=img_std)
else:
    normalize = []

celeba_transform_test = transforms.Compose([
    transforms.ToTensor()] + normalize)

test_split = 'test' if args.test else 'valid'

celeba_test_dataset = datasets.CelebA(args.dataset_path, split=test_split,
    target_type='attr', download=False, transform=celeba_transform_test)

batch_size = args.batch_size

test_loader = torch.utils.data.DataLoader(celeba_test_dataset, batch_size,
    shuffle=False, num_workers=2, pin_memory=True)


checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

resume_file = f'{checkpoint_dir}/{args.resume}'
assert os.path.isfile(resume_file)
checkpoint = torch.load(resume_file)

INPUT_CHANNELS = checkpoint.get('input_channels', 3)

assert INPUT_CHANNELS in (3, 6)

USE_6_CHANNELS = (INPUT_CHANNELS == 6)

model = resnet.resnet50(num_classes=40,
                        input_channels=INPUT_CHANNELS)

if checkpoint['normalize']:
    if args.normalize:
        print('Warning: both early and late normalizations are active')
    model = nn.Sequential([NormalizeLayer(img_mean, img_std), model])

model = model.to(device)

if CUDA and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)
model.load_state_dict(checkpoint['net'])

# writer = SummaryWriter(log_dir=args.log_dir)

correct = total = 0

for X, y in tqdm.tqdm(test_loader):
    total += len(y)

    y_pred = (model(X) > 0).type(torch.int)

    correct += (y_pred == y).sum().item()


acc = 100. * correct / total
print('Results: ')
print(f'Accuracy {acc:.1f}% ({correct})')

if args.rewrite_accuracy:
    checkpoint['acc'] = acc
    torch.save(checkpoint, resume_file)
