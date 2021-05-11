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
                      AdversarialAttackAugmenter, RandomNoiseAugmenter,
                      calculate_radii, get_certified_accuracies, NormalizeLayer,
                      yes_or_no_p)

parser = argparse.ArgumentParser(description='PyTorch Celeba Training')
parser.add_argument('--lr', default=1e-3, type=float, help='learning rate')
parser.add_argument('--batch-size', default=64, type=int, help='batch size')
parser.add_argument('--normalize', action='store_true',
                    help='use normalization layer')
parser.add_argument('--regularization', default=5e-4, type=float,
                    help='weight decay')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--seed', default=1, type=int, help='seed')
parser.add_argument('--end-epoch', default=50, type=int, help='end epoch')
parser.add_argument('--dataset-path', default='dsets/',
                    help='path to CelebA dataset')
parser.add_argument('--log-dir', default='runs/', help='Tensorboard directory')
parser.add_argument('--adv-steps', default=2, type=int, help='steps of PGD')
parser.add_argument('--epsilon', default=1.0, type=float, help='max PGD norm')
parser.add_argument('--noise-cnt', default=2, type=int, help='number of noises')
parser.add_argument('--sigma', default=1.0, type=float, help='scale of noise')
parser.add_argument('-y', action='store_true', help='force overwrite')

args = parser.parse_args()


torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)

checkpoint_dir = 'checkpoints'
if not os.path.exists('./checkpoints'):
    os.makedirs('./checkpoints')
norm_tok = '_norm' if args.normalize else ''
checkpoint_file = f'./{checkpoint_dir}/adv-lr_{args.lr}_decay_{args.regularization}{norm_tok}_epoch_{{}}.pth'

CUDA = torch.cuda.is_available()
if CUDA:
    device = torch.device('cuda')
    print(torch.cuda.get_device_name(device))
    cudnn.benchmark = True
else:
    device = torch.device('cpu')

start_epoch = 1

img_mean = [0.5063, 0.4258, 0.3832]
img_std = [0.2660, 0.2452, 0.2414]

celeba_transform_test = transforms.Compose([
    transforms.ToTensor(),
])

celeba_transform_train = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
])

celeba_train_dataset = datasets.CelebA(args.dataset_path, split='train',
    target_type='attr', download=False, transform=celeba_transform_train)

celeba_test_dataset = datasets.CelebA(args.dataset_path, split='valid',
    target_type='attr', download=False, transform=celeba_transform_test)

batch_size = args.batch_size

assert batch_size > 0

train_loader = torch.utils.data.DataLoader(celeba_train_dataset, batch_size,
    shuffle=True, num_workers=2, pin_memory=True)
test_loader = torch.utils.data.DataLoader(celeba_test_dataset, batch_size * 2,
    shuffle=False, num_workers=2, pin_memory=True)

model = resnet.resnet50(num_classes=40, input_channels=3)

if args.normalize:
    model = nn.Sequential([NormalizeLayer(img_mean, img_std), model])

model = model.to(device)

criterion = nn.BCEWithLogitsLoss()

if args.resume:
    print('Resuming....')
    assert os.path.isdir(checkpoint_dir), 'No checkpoint directory found!'
    resume_file = f'{checkpoint_dir}/{args.resume}'
    assert os.path.isfile(resume_file)
    checkpoint = torch.load(resume_file)
    model.load_state_dict(checkpoint['net'])
    start_epoch = checkpoint['epoch']+1
    assert args.normalize == checkpoint['normalize']
    checkpoint_file = f'./{checkpoint_dir}/adv-lr_{args.lr}_decay_{args.regularization}{norm_tok}_epoch_{{}}_resume_{args.resume}.pth'

if os.path.isfile(checkpoint_file.format(start_epoch)):
    if (not args.y and
        not yes_or_no_p('Checkpoint file already exists, overwrite?')):
        exit()

if CUDA and torch.cuda.device_count() > 1:
    model = nn.DataParallel(model)

optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=.9,
                            weight_decay=args.regularization)

if args.adv_steps:
    augmenter = AdversarialAttackAugmenter(args.adv_steps, args.epsilon,
                                           args.noise_cnt, args.sigma)
else:
    augmenter = RandomNoiseAugmenter(args.sigma)

writer = SummaryWriter(log_dir=args.log_dir)

def train(epoch):
    assert model.training
    loss_cnt = MovingAverageCounter(5)
    iteration = (epoch - 1) * len(train_loader.dataset)

    for X, y in tqdm.tqdm(train_loader):
        input = X.to(device)
        target = y.to(device).to(torch.float32)
        X_aug, y_aug = augmenter.augment_tensors(input, target, model)
        y_pred = model(X_aug)
        loss = criterion(y_pred, y_aug)
        loss_cnt.add(loss.item())
        iteration += len(y)
        writer.add_scalar('Loss/train', loss_cnt.get_average(), iteration)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def validate(epoch):
    assert model.training
    try:
        model.eval()
        loss_cnt = AverageCounter()
        correct = 0
        total = 0
        iteration = (epoch - 1) * len(test_loader.dataset)
        with torch.no_grad():
            for X, y in tqdm.tqdm(test_loader):
                iteration += len(y)

                y_pred = model(X.to(device))
                y = y.to(device)
                correct += torch.sum((y_pred > 0) == y).item()
                total += y_pred.numel()
                loss = criterion(y_pred, y.to(y_pred.dtype))
                loss_cnt.add(loss.item())
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
        'model': 'resnet50',
        'normalize': args.normalize,
    }
    if not os.path.isdir(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    torch.save(state, checkpoint_file.format(epoch))
    torch.save(state, "checkpoints/adv_" + str(epoch)+".pth")

def small_certify(epoch, k=25):
    def pl(ax, d, **kwargs):
        radii, res = get_certified_accuracies(d)
        ax.plot(radii, res, **kwargs)
    res = calculate_radii(test_loader.dataset, range(k), model, 10**2, 10**4,
        1e-3, args.sigma, device=device, batch_size=batch_size*2)

    fig, ax = plt.subplots()
    pl(ax, res, label='certified accuracy')
    ax.axhline(0.818, label='constant', c='k')
    ax.legend()
    writer.add_figure('certified-accuracy', fig, epoch)

    cert_dir = os.path.join(checkpoint_dir, 'certs')
    if not os.path.isdir(cert_dir):
        os.makedirs(cert_dir)

    npath_base, npath_name = os.path.split(checkpoint_file.format(epoch))
    npath = os.path.join(npath_base, 'certs', npath_name)
    torch.save(res, npath)

for epoch in range(start_epoch,  args.end_epoch+1):
    print(f'Epoch {epoch:3d}')
    train(epoch)
    validate(epoch)
    small_certify(epoch, 25)
