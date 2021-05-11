#/usr/bin/env python3

from collections import deque
import time
import torch
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import numpy as np
import torchvision
from torchvision import datasets, transforms, models
import tqdm

from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint

class RepeatableTimer:
    def __init__(self, secs, up_initially=False):
        self.secs = secs
        if up_initially:
            self.last_time = -secs
        else:
            self.last_time = time.time()

    def check_time(self):
        cur_time = time.time()
        if self.last_time + self.secs < cur_time:
            self.last_time = cur_time
            return True
        else:
            return False


class PeriodicPrinter:
    def __init__(self, secs):
        self.timer = RepeatableTimer(secs, up_initially=True)

    def print(self, *args):
        up_already = self.timer.check_time()
        if up_already:
            print(*args)
        return up_already


class AverageCounter:
    def __init__(self):
        self.value = 0.
        self.total = 0

    def get_average(self):
        if self.total == 0:
            return float('nan')
        else:
            return self.value / self.total

    def add(self, x):
        self.total += 1
        self.value += x


class MovingAverageCounter:
    def __init__(self, N):
        self.d = deque()
        self.max_size = N
        self.sum = 0.
        self.time_since_recalculation = 0

    def add(self, x):
        self.d.appendleft(x)
        self.sum += x
        if len(self.d) > self.max_size:
            self.sum -= self.d.pop()
            self.time_since_recalculation += 1
        if self.time_since_recalculation >= 100 * self.max_size:
            self._recalculate()

    def _recalculate(self):
        self.sum = sum(self.d)
        self.time_since_recalculation = 0

    def get_average(self):
        if self.d:
            return self.sum / len(self.d)
        else:
            return float('nan')


@torch.no_grad()
def get_prediction_counts(model, x, mc_size, batch_size, sigma, xrange=range):
    counts = None
    for processed in xrange(0, mc_size, batch_size):
        sz = min(batch_size, mc_size - processed)
        noised_inputs = x.expand(sz, *x.shape)
        noised_inputs = noised_inputs + torch.randn_like(noised_inputs) * sigma
        outputs = (model(noised_inputs) > 0).to(torch.int64)
        if counts is None:
            counts = torch.zeros(2, outputs.shape[1], dtype=torch.int64,
                                 device=outputs.device)
        add = outputs.sum(dim=0)
        counts[1] += add
        counts[0] += sz - add
    return counts


def certify(model, x, n_sel, n_est, alpha, sigma, batch_size, xrange=range):
    """ Certify radius with probability 1 - alpha
    :param x: the input [channel x height x width]
    :param n_sel: the number of Monte Carlo samples to use for selection
    :param n_est: the number of Monte Carlo samples to use for estimation
    :param alpha: the error probability
    :return: (predicted classes, certified radii), if the prediction was rejected, the predicted class is 2 and radius 0
    """

    counts_selection = get_prediction_counts(model, x, n_sel, batch_size, sigma, xrange=xrange)
    predicted_class = (counts_selection[1] > counts_selection[0]).to(torch.int64)
    counts_est = get_prediction_counts(model, x, n_est, batch_size, sigma, xrange=xrange)

    result_class = []
    result_radius = []

    for i in range(len(predicted_class)):
        count_est = counts_est[predicted_class[i].item(),i].item()
        est_lo, est_hi = proportion_confint(count_est, n_est, 2 * alpha, method='beta')
        if est_lo < 0.5:
            result_class.append(2)  # abstain
            result_radius.append(0.)
        else:
            result_class.append(predicted_class[i].item())
            result_radius.append(sigma * norm.ppf(est_lo))

    return torch.tensor(result_class), torch.tensor(result_radius)


def certified_accuracy(pred_cl, pred_radius, target_cl, threshold_radius):
    return torch.mean(((pred_cl == target_cl) & (pred_radius >= threshold_radius)).to(torch.float)).item()


def calculate_radii(dataset, indicies, model, n_sel, n_est, alpha, sigma, device=torch.device('cpu'), batch_size=64):
    predicted_classes = []
    targets = []
    predicted_radii = []
    for i in tqdm.tqdm(indicies):
        img, target = dataset[i]
        res_cl, res_rad = certify(model, img.to(device), n_sel, n_est, alpha, sigma, batch_size)
        predicted_classes.append(res_cl)
        predicted_radii.append(res_rad)
        targets.append(target)
    predicted_classes = torch.stack(predicted_classes)
    targets = torch.stack(targets)
    predicted_radii = torch.stack(predicted_radii)

    return {'y': targets, 'y_pred': predicted_classes, 'r': predicted_radii}


def get_certified_accuracies(d):
    radii = np.linspace(0, d['r'].view(-1).max())
    res = np.zeros_like(radii)
    for i, r in enumerate(radii):
        res[i] = certified_accuracy(d['y_pred'].view(-1), d['r'].view(-1), d['y'].view(-1), r)
    return radii, res


class NoAugmenter:
    def augment_tensors(self, X, y, model):
        return X, y


class RandomNoiseAugmenter:
    def __init__(self, sigma):
        self.sigma = sigma

    def augment_tensors(self, X, y, model):
        X = X + torch.randn_like(X, device=X.device) * self.sigma
        return X, y

class AdversarialAttackAugmenter:
    def __init__(self, steps, max_norm, noise_cnt, sigma):
        self.steps = steps
        self.max_norm = max_norm
        self.noise_cnt = noise_cnt
        self.sigma = sigma

    def augment_tensors(self, X, y, model, return_noises=False):
        assert model.training
        X = X.repeat(self.noise_cnt, 1, 1, 1)
        noise = torch.randn_like(X, device=X.device) * self.sigma
        try:
            for param in model.parameters():
                assert param.requires_grad
                param.requires_grad_(False)
            model.eval()
            X = self._attack(model, X, y, noise)
        finally:
            model.train()
            for param in model.parameters():
                param.requires_grad_(True)
        y = y.repeat(self.noise_cnt, 1)
        X.add_(noise)

        if return_noises:
            return X, y, noises
        else:
            return X, y

    def _attack(self, model, X, y, noise):
        batch_size = y.shape[0]
        delta = torch.zeros(batch_size, *X.shape[1:], device=X.device, requires_grad=True)
        optimizer = torch.optim.SGD([delta], lr=self.max_norm/self.steps*2)
        for i in range(self.steps):
            X_adv = X + delta.repeat(self.noise_cnt, 1, 1, 1) + noise
            logits = model(X_adv)
            probs = torch.sigmoid(logits).view(self.noise_cnt, batch_size, logits.shape[-1]).mean(dim=0)

            loss = -torch.nn.functional.binary_cross_entropy(probs, y)
            optimizer.zero_grad()
            loss.backward()

            grad_norms = delta.grad.view(batch_size, -1).norm(p=2, dim=1)
            delta.grad.div_(grad_norms.view(-1, 1, 1, 1))
            if (grad_norms == 0).any():
                delta.grad[grad_norms == 0] = torch.randn_like(delta.grad[grad_norms == 0], device=delta.device)

            optimizer.step()

            delta.data.add_(X[:batch_size])
            delta.data.clamp_(0, 1).sub_(X[:batch_size])

            delta.data.renorm_(p=2, dim=0, maxnorm=self.max_norm),
        return X + delta.repeat(self.noise_cnt, 1, 1, 1)


def mask_batch(batch, pos, block_size):
    bs, ch, h, w = batch.shape
    assert 0 <= pos < w
    out_c1 = torch.zeros_like(batch)
    out_c2 = torch.zeros_like(batch)
    if pos + block_size > w:
        out_c1[:,:,:,pos:] = batch[:,:,:,pos:]
        out_c2[:,:,:,pos:] = 1. - batch[:,:,:,pos:]

        rem = block_size - (w - pos)
        out_c1[:,:,:,:rem] = batch[:,:,:,:rem]
        out_c2[:,:,:,:rem] = 1. - batch[:,:,:,:rem]
    else:
        out_c1[:,:,:, pos:pos+block_size] = batch[:,:,:,pos:pos+block_size]
        out_c2[:,:,:, pos:pos+block_size] = 1. - batch[:,:,:,pos:pos+block_size]
    return torch.cat([out_c1, out_c2], 1)


def random_mask_batch(batch, block_size):
    pos = torch.randint(batch.size(3), size=(1,)).item()
    return mask_batch(batch, pos, block_size)

def random_mask_batch_alt(batch, block_size):
    batch = batch.permute(0, 2, 3, 1) # color channel last
    bs, h, w, ch = batch.shape
    out_c1 = torch.zeros_like(batch)
    out_c2 = torch.zeros_like(batch)
    pos = torch.randint(w, size=(1,)).item()
    if pos + block_size > w:
        out_c1[:,:,pos:] = batch[:,:,pos:]
        out_c2[:,:,pos:] = 1. - batch[:,:,pos:]

        out_c1[:,:,:pos+block_size-w] = batch[:,:,:pos+block_size-w]
        out_c2[:,:,:pos+block_size-w] = 1. - batch[:,:,:pos+block_size-w]
    else:
        out_c1[:,:,pos:pos+block_size] = batch[:,:,pos:pos+block_size]
        out_c2[:,:,pos:pos+block_size] = 1. - batch[:,:,pos:pos+block_size]
    out_c1 = out_c1.permute(0, 3, 1, 2)
    out_c2 = out_c2.permute(0, 3, 1, 2)
    out = torch.cat((out_c1,out_c2), 1)
    #print(out[14,:,5:10,5:10])
    return out

def yes_or_no_p(prompt):
    while True:
        txt = input(f'{prompt} [y/n]: ')
        if txt.lower().startswith('y'):
            return True
        elif txt.lower().startswith('n'):
            return False
        else:
            print("Please enter 'Yes' or 'No' ")


class NormalizeLayer(torch.nn.Module):
    """Standardize the channels of a batch of images by subtracting the dataset mean
      and dividing by the dataset standard deviation.
      In order to certify radii in original coordinates rather than standardized coordinates, we
      add the Gaussian noise _before_ standardizing, which is why we have standardization be the first
      layer of the classifier rather than as a part of preprocessing as is typical.
      """

    def __init__(self, means, stds):
        """
        :param means: the channel means
        :param stds: the channel standard deviations
        """
        super(NormalizeLayer, self).__init__()
        self.register_buffer('means', torch.tensor(means), persistent=False)
        self.register_buffer('stds', torch.tensor(stds), persistent=False)

    def forward(self, x):
        means = self.means.view(1, -1, 1, 1).expand_as(x)
        stds = self.std.view(1, -1, 1, 1).expand_as(x)
        return (x - means) / sds


class NullWriter:
    def add_scalar(tag, scalar_value, global_step=None, walltime=None):
        pass

    def add_scalars(main_tag, tag_scalar_dict, global_step=None, walltime=None):
        pass

    def add_figure(tag, figure, global_step=None, close=True, walltime=None):
        pass


CELEBA_IMG_MEAN = [0.5063, 0.4258, 0.3832]
CELEBA_IMG_STD  = [0.2660, 0.2452, 0.2414]

def get_train_dataset(dataset_path, normalize=False):
    dset_transforms = [
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
    ]
    if normalize:
        dset_transforms.append(transforms.Normalize(mean=CELEBA_IMG_MEAN,
                                                    std=CELEBA_IMG_STD))
    dset_transforms = transforms.Compose(dset_transforms)
    return datasets.CelebA(args.dataset_path, split='train',
        target_type='attr', download=False, transform=dset_transforms)


def get_test_dataset(dataset_path, normalize=False, test_set=False):
    if normalize:
        celeba_transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=CELEBA_IMG_MEAN, std=CELEBA_IMG_STD),
        ])
    else:
        celeba_transform_test = transforms.ToTensor()

    test_split = 'test' if test_set else 'valid'

    return datasets.CelebA(dataset_path, split=test_split,
        target_type='attr', download=False, transform=celeba_transform_test)


""" Some statistical functions to allow differentiable binary proportion estimation """
def erfcinv(x):
    return torch.erfinv(1 - x)

def ppf(x):
    return - 2**.5 * erfcinv(2 * x)

def wilson(ns, n, alpha=0.01):
    z = norm.ppf(alpha) * 0.5
    p_cent = (ns + 0.5*z**2) / (n + z**2)
    p_disc_sqrt = z / (n + z**2) * (ns*(n - ns)/n+z**2 * 0.25)**.5
    return (p_cent + p_disc_sqrt, p_cent - p_disc_sqrt)

def wilson_estimate_radii(y_pred, y_true, alpha, sigma):
    # Assuming y_pred: [bs x classes x n_est]

    batch_size, classes, n_est = y_pred.shape

    result = torch.zeros(batch_size, classes)
    y_true = y_true.unsqueeze(-1).expand_as(y_pred)
    y_pred_alt = y_pred * y_true + (1 - y_pred) * (1 - y_true)
    est_lo, est_hi = wilson(y_pred_alt.sum(dim=-1), n_est, alpha=2*alpha)
    CLAMP_LO = 1e-9
    CLAMP_HI = 1 - 1e-9
    clamped = (est_lo < CLAMP_LO) | (est_lo > CLAMP_HI)
    return sigma * ppf(est_lo.type(torch.float64).clamp(min=CLAMP_LO, max=CLAMP_HI)), clamped



if __name__ == '__main__':
    raise NotImplemented('this module is not for launching')
