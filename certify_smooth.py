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
import matplotlib.pyplot as plt

from smoothing import (AverageCounter, MovingAverageCounter,
                      calculate_radii, get_certified_accuracies,
                      get_test_dataset, yes_or_no_p)

parser = argparse.ArgumentParser(description='PyTorch Celeba Certification')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--checkpoint', help='checkpoint')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--log-dir', default='runs/', help='Tensorboard directory')
parser.add_argument('--noise-cnt', default=2, type=int, help='number of noises')
parser.add_argument('--sigma', default=1.0, type=float, help='scale of noise')
parser.add_argument('--wilson', action='store_true',
                    help='Use Wilson interval instead of Clopper-Pearson')
#parser.add_argument('--radii', default='0,0.5,1.0,1.5,2.0,2.5,3.0',
#                    help='comma-separated list of radii to consider')
parser.add_argument('--k', type=int, default=None,
                    help='estimate based on K images only')
parser.add_argument('--n-select', type=int, default=100,
                    help='How many samples to use to estimate class')

parser.add_argument('--n-estimate', type=int, default=10000,
                    help='How many samples to use to estimate radius')
parser.add_argument('--test', action='store_true',
                    help='Use test set instead of validation')
parser.add_argument('--figure', help='where to store the plot')
parser.add_argument('--radii', help='where to store the radii', required=True)

args = parser.parse_args()

torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

CUDA = torch.cuda.is_available()

if CUDA:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))
    cudnn.benchmark = True
else:
    device = torch.device('cpu')


batch_size = args.batch_size

assert batch_size > 0

assert os.path.isfile(checkpoint), f'chekpoint does not exist {checkpoint}'
checkpoint = torch.load(checkpoint)
celeba_test_dataset = get_test_dataset(args.dataset_path,
                                       normalize=checkpoint['normalize'],
                                       test_set=args.test)

model = resnet.resnet50(num_classes=40, input_channels=3)

if checkpoint['normalize']:
    model = nn.Sequential([NormalizeLayer(img_mean, img_std), model])
model = model.to(device)
model = torch.DataParallel(model)

model.load_state_dict(checkpoint['net'])

if args.k:
    rng = range(k)
else:
    rng = range(len(celeba_test_dataset))

radii = calculate_radii(celeba_test_dataset, rng, model, args.n_select,
                        args.n_estimate, args.sigma, device=device,
                        batch_size=batch_size)

if args.radii:
    torch.save(radii, args.radii)

if args.figure:
    fig, ax = plt.subplots()
    rad, res = get_certified_accuracies(radii)
    ax.plot(rad, res)
    fig.savefig(args.figure)
