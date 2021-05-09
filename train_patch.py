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
                       random_mask_batch, yes_or_no_p)


parser = argparse.ArgumentParser(description='PyTorch Celeba Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--band-size', default=10, type=int, help='band size')
parser.add_argument('--regularization', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--end-epoch', default=50, type=int, help='end epoch')
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--log-dir', default='runs/', help='Tensorboard directory')

args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
checkpoint_file = f'./{checkpoint_dir}/pb-lr_{args.lr}_decay_{args.regularization}__band_{args.band_size}_epoch_{{}}.pth'

print(f'checkpoint file is {checkpoint_file}')

CUDA = torch.cuda.is_available()

if CUDA:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))
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

batch_size = 128

train_loader = torch.utils.data.DataLoader(celeba_train_dataset, batch_size,
    shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(celeba_test_dataset, batch_size,
    shuffle=False, num_workers=2, pin_memory=True)

model = resnet.resnet50(num_classes=40, input_channels=6).to(device)

if CUDA:
    model = nn.DataParallel(model)
    cudnn.benchmark = True

criterion = nn.BCEWithLogitsLoss()

if args.resume:
    print('Resuming....')
    assert os.path.isdir(checkpoint_dir), 'No checkpoint directory found!'
    resume_file = f'{checkpoint_dir}/{args.resume}'
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    checkpoint_file = f'./{checkpoint_dir}/pb-lr_{args.lr}_decay_{args.regularization}__band_{args.band_size}_epoch_{{}}_resume_{args.resume}.pth'

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
    loss_acc = 0
    correct_acc = 0
    total_samples = 0
    total_classes = 0
    iteration = (epoch - 1) * len(train_loader.dataset)

    for i, (X, y) in enumerate(tqdm.tqdm(train_loader)):
        iteration += len(y)
        X = X.to(device)
        y = y.to(device).to(torch.float32)

        y_pred = model(random_mask_batch(X, args.band_size))
        loss = criterion(y_pred, y)
        loss_cnt.add(loss.item())

        writer.add_scalar('Loss/train', loss_cnt.get_average(), iteration)
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

            y_pred = model(random_mask_batch(X, args.band_size))

            correct += torch.sum((y_pred > 0) == y).item()
            loss = criterion(y_pred, y.to(y_pred.dtype))
            loss_cnt.add(loss.item())
            total += y_pred.numel()
            loss_cur = loss_cnt.get_average()
            writer.add_scalar('Loss/val-it', loss_cur, iteration)
            writer.add_scalar('Accuracy/val-it', correct / total, iteration)
    finally:
        model.train()

    acc = 100. * correct / total
    print('Saving..')
    state = {
        'net': model.state_dict(),
        'acc': acc,
        'epoch': epoch,
        'band_size': args.band_size,
        'input_channels': 6,
    }
    if not os.path.isdir(checkpoint_dir):
        os.mkdir(checkpoint_dir)
    torch.save(state, checkpoint_file.format(epoch))
    torch.save(state, "checkpoints/pb_" + str(epoch)+".pth")


for epoch in range(start_epoch,  args.end_epoch+1):
    train(epoch)
    validate(epoch)
