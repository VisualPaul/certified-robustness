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
                      get_test_dataset, yes_or_no_p, NormalizeLayer)

parser = argparse.ArgumentParser(description='PyTorch Celeba Certification')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--checkpoint', help='checkpoint', required=True)
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--log-dir', default='runs/', help='Tensorboard directory')
parser.add_argument('--sigma', default=1.0, type=float, help='scale of noise')
parser.add_argument('--alpha', default=0.01, type=float, help='confidence level')
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
parser.add_argument('--normalize', action='store_true',
                    help='normalize dataset')
parser.add_argument('--test', action='store_true',
                    help='Use test set instead of validation')
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

celeba_test_dataset = get_test_dataset(args.dataset_path,
                                       normalize=args.normalize,
                                       test_set=args.test)

model = resnet.resnet50(num_classes=40, input_channels=3)

img_mean = [0.5063, 0.4258, 0.3832]
img_std = [0.2660, 0.2452, 0.2414]

# if checkpoint.get('normalize', False):
#     model = nn.Sequential([NormalizeLayer(img_mean, img_std), model])
model = model.to(device)
#model = nn.DataParallel(model)

for i in tqdm.trange(1, 51):
    checkpoint = args.checkpoint.format(num=i)
    radii_name = args.radii.format(num=i)

    print(f'checkpoint is "{checkpoint}", saving as "{radii_name}"')

    assert os.path.isfile(checkpoint), f'chekpoint does not exist {checkpoint}'
    checkpoint = torch.load(checkpoint)

    model.load_state_dict(checkpoint['net'])

    if args.k:
        rng = range(args.k)
    else:
        rng = range(len(celeba_test_dataset))

    if i == 1:
        rng = tqdm.tqdm(rng)

    radii = calculate_radii(celeba_test_dataset, rng, model, args.n_select,
                            args.n_estimate, args.alpha, args.sigma, device=device,
                            batch_size=batch_size)

    torch.save(radii, radii_name)
