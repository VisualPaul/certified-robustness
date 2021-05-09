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


parser = argparse.ArgumentParser(description='PyTorch Celeba Training')
parser.add_argument('--band-size', default=10, type=int, help='band size')
parser.add_argument('--certify-size', default=20, type=int,
                    help='size to certify')
parser.add_argument('--threshold', default=0.1, type=float,
                    help='threshold for smoothing abstain')

parser.add_argument('--batch-size', type=int, default=128, help='batch size')
parser.add_argument('--checkpoint', help='checkpoint', required=True)
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--test', action='store_true',
                    help='Use test set instead of validation')

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

normalize = transforms.Normalize(mean=img_mean, std=img_std)

celeba_transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

test_split = 'test' if args.test else 'valid'

celeba_test_dataset = datasets.CelebA(args.dataset_path, split=test_split,
    target_type='attr', download=False, transform=celeba_transform_test)

batch_size = args.batch_size

test_loader = torch.utils.data.DataLoader(celeba_test_dataset, batch_size,
    shuffle=False, num_workers=2, pin_memory=True)

model = resnet.resnet50(num_classes=40, input_channels=6).to(device)

model = nn.DataParallel(model)

criterion = nn.BCEWithLogitsLoss()

checkpoint_dir = './checkpoints'
if not os.path.exists(checkpoint_dir):
    os.makedirs(checkpoint_dir)

print('Resuming....')
checkpoint_file = f'{checkpoint_dir}/{args.checkpoint}'
assert os.path.isfile(checkpoint_file)
checkpoint = torch.load(checkpoint_file)
model.load_state_dict(checkpoint['net'])
band_size = checkpoint.get('band_size', 10)

# writer = SummaryWriter(log_dir=args.log_dir)

@torch.no_grad()
def certify_patch(batch):
    predictions_0 = torch.zeros(X.shape[0], 40, device=device, dtype=torch.int)
    predictions_1 = torch.zeros(X.shape[0], 40, device=device, dtype=torch.int)
    bs, ch, h, w = batch.shape

    for pos in range(w):
        out = torch.sigmoid(model(mask_batch(batch, pos, band_size)))
        predictions_1 += (out > 0.5 + args.threshold).type(torch.int)
        predictions_0 += (out < 0.5 - args.threshold).type(torch.int)
    num_affected = args.certify_size + args.band_size - 1
    predictions = (predictions_1 > predictions_0).type(torch.int)

    predictions_diff = (torch.maximum(predictions_0, predictions_1) -
                        torch.minimum(predictions_0, predictions_1))

    cert = ((predictions_diff > 2 * num_affected) |
            (predictions_diff == 2 * num_affected) & (predictions == 0))

    return predictions, cert


correct = cert_correct = certified = total = 0

for X, y in tqdm.tqdm(test_loader):
    total += y.num_elements()

    X = X.to(device)
    y = y.to(device)

    y_pred, cert = certify_patch(X)

    correct += (y_pred == y).sum().item()
    certified += cert.sum().item()
    cert_correct += ((y_pred == y) & cert).sum().item()

print('Parameters: ')
print(f'Band size is {args.band_size}')
print(f'Certifying against {args.certify_size}x{args.certify_size}')
print(f'Using threshold {args.threshold}')
print('Results: ')
print(f'Certified accuracy {cert_correct / total * 100.:.1f}% ({cert_correct})')
print(f'Accuracy {correct / total * 100.:.1f}% ({correct})')
print(f'Certified percentage {certified / total * 100.:.1f}% ({certified})')
print(f'Images total {total}')
