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
                      AdversarialAttackAugmenter, calculate_radii,
                      get_certified_accuracies, NormalizeLayer, yes_or_no_p)

parser = argparse.ArgumentParser(description='PyTorch Celeba Training')
parser.add_argument('--checkpoint', required=True, help='checkpoint')
parser.add_argument('--batch-size', default=128, type=int, help='batch size')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--log-dir', default='runs/', help='Tensorboard directory')
parser.add_argument('--sigma', default=1.0, type=float, help='scale of noise')

args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = 'checkpoints'
CUDA = torch.cuda.is_available()
if CUDA:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

img_mean = [0.5063, 0.4258, 0.3832]
img_std = [0.2660, 0.2452, 0.2414]

celeba_transform_test = transforms.Compose([
    transforms.ToTensor(),
])

celeba_test_dataset = datasets.CelebA(args.dataset_path, split='valid',
    target_type='attr', download=False, transform=celeba_transform_test)

batch_size = args.batch_size

assert batch_size > 0

test_loader = torch.utils.data.DataLoader(celeba_test_dataset, batch_size * 2,
    shuffle=False, num_workers=2, pin_memory=True)

model = resnet.resnet50(num_classes=40, input_channels=3)

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

assert os.path.isdir(checkpoint_dir), 'No checkpoint directory found!'
checkpoint_file = f'./{checkpoint_dir}/{args.checkpoint}'
assert os.path.isfile(checkpoint_file)
checkpoint = torch.load(checkpoint_file)
if checkpoint['normalize']:
    model = nn.Sequential([NormalizeLayer(img_mean, img_std), model])
epoch = checkpoint['epoch']
model.load_state_dict(checkpoint['net'])



if CUDA and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

writer = SummaryWriter(log_dir=args.log_dir)

def small_certify(epoch, k=25):
    def pl(ax, d, **kwargs):
        radii, res = get_certified_accuracies(d)
        ax.plot(radii, res, **kwargs)
    res = calculate_radii(test_loader.dataset, range(k), model, 100, 10**4,
        1e-3, args.sigma, device=device, batch_size=batch_size*2)

    fig, ax = plt.subplots()
    pl(ax, res, label='certified accuracy')
    ax.axhline(0.818, label='constant', c='k')
    ax.legend()
    writer.add_figure('certified-accuracy', fig, epoch)

    cert_dir = os.path.join(checkpoint_dir, 'certs')
    if not os.path.isdir(cert_dir):
        os.makedirs(cert_dir)

    npath_base, npath_name = os.path.split(checkpoint_file)
    npath = os.path.join(npath_base, 'certs', npath_name)
    torch.save(res, npath)

small_certify(epoch, 25)
