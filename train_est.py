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

from smoothing import (wilson_estimate_radii, yes_or_no_p, MovingAverageCounter,
                       AverageCounter)

parser = argparse.ArgumentParser(description='PyTorch Celeba Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--regularization', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--end-epoch', default=50, type=int, help='end epoch')
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--log-dir', default='runs/', help='Tensorboard directory')
parser.add_argument('--num-estimators', default=5, type=int,
                    help='number of examples to use for radius estimation')
parser.add_argument('--alpha', default=0.01, type=float,
                    help='confidence level for the interval')
parser.add_argument('--sigma', default=1.0, type=float,
                    help='noise scale')
parser.add_argument('-y', action='store_true', help='force overwrite')
parser.add_argument('--cpu', action='store_true', help='force CPU')

args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_file = f'./{checkpoint_dir}/rad-lr_{args.lr}_decay_{args.regularization}_epoch_{{}}.pth'

print(f'checkpoint file is {checkpoint_file}', file=sys.stderr)

CUDA = torch.cuda.is_available() and not args.cpu

if CUDA:
    device = torch.device('cuda')
    print(f'CUDA_VISIBLE_DEVICES={os.environ["CUDA_VISIBLE_DEVICES"]}', file=sys.stderr)
    print(torch.cuda.get_device_name(device), file=sys.stderr)
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

start_epoch = 1

img_mean = [0.5063, 0.4258, 0.3832]
img_std = [0.2660, 0.2452, 0.2414]

normalize = transforms.Normalize(mean=img_mean, std=img_std)

celeba_transform_test = transforms.Compose([
    transforms.ToTensor(),
    normalize,
])

celeba_transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    normalize,
])

celeba_train_dataset = datasets.CelebA(args.dataset_path, split='train',
    target_type='attr', download=False, transform=celeba_transform_train)

celeba_test_dataset = datasets.CelebA(args.dataset_path, split='valid',
    target_type='attr', download=False, transform=celeba_transform_test)

batch_size = args.batch_size

train_loader = torch.utils.data.DataLoader(celeba_train_dataset, batch_size,
    shuffle=True, num_workers=2, pin_memory=CUDA)
test_loader = torch.utils.data.DataLoader(celeba_test_dataset, batch_size,
    shuffle=False, num_workers=2, pin_memory=CUDA)

model = resnet.resnet50(num_classes=40, input_channels=3).to(device)

criterion = nn.BCEWithLogitsLoss()

if args.resume:
    print('Resuming....', file=sys.stderr)
    assert os.path.isdir(checkpoint_dir), 'No checkpoint directory found!'
    assert os.path.isfile(args.resume)
    checkpoint = torch.load(args.resume)
    resume_name = os.path.basename(args.resume)
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    checkpoint_file = f'./{checkpoint_dir}/rad-lr_{args.lr}_decay_{args.regularization}_epoch_{{}}_resume_{resume_name}.pth'

if os.path.isfile(checkpoint_file.format(start_epoch)):
    if not yes_or_no_p('Checkpoint file already exists, overwrite?'):
        exit()

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9,
                            weight_decay=args.regularization)

writer = SummaryWriter(log_dir=args.log_dir)

def train(epoch):
    print(f'\nEpoch {epoch}')
    assert model.training
    loss_cnt = MovingAverageCounter(5)
    clamped_cnt = MovingAverageCounter(5)
    loss_acc = 0
    correct_acc = 0
    total_samples = 0
    total_classes = 0
    iteration = (epoch - 1) * len(train_loader.dataset)

    for i, (X, y) in enumerate(tqdm.tqdm(train_loader)):
        iteration += len(y)
        X = X.to(device)
        y = y.to(device).to(torch.float32)

        y_pred = torch.zeros(*y.shape, args.num_estimators, device=device)
        for j in range(args.num_estimators):
            y_pred[:,:,j] = model(X + torch.randn_like(X) * args.sigma)
        y_pred = torch.sigmoid(y_pred)
        r, c = wilson_estimate_radii(y_pred, y, alpha=args.alpha,
                                     sigma=args.sigma)
        loss = -r.mean()
        loss_cnt.add(loss.item())
        clamped_cnt.add(c.type(torch.float64).mean().item())
        writer.add_scalar('Loss/train', loss_cnt.get_average(), iteration)
        writer.add_scalar('Clamped/train', clamped.get_average(), iteration)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


@torch.no_grad()
def validate(epoch):
    loss_cnt = AverageCounter()
    correct = 0
    total = 0
    try:
        model.eval()
        iteration = (epoch - 1) * len(test_loader.dataset)
        for X, y in tqdm.tqdm(test_loader):
            iteration += len(y)
            X = X.to(device)
            y = y.to(device)

            y_pred = model(X)
            total += y.numel()

            correct += torch.sum((y_pred > 0) == y).item()
            writer.add_scalar('Accuracy/val-it', correct / total, iteration)
    finally:
        model.train()

    acc = 100. * correct / total
    print('Saving..', file=sys.stderr)
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'input_channels': 3,
        'normalize': True,
    }
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_file.format(epoch))
    torch.save(state, "checkpoints/rad_" + str(epoch)+".pth")


for epoch in range(start_epoch,  args.end_epoch+1):
    train(epoch)
    validate(epoch)
