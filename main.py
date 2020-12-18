import torch
from torch import functional as F
from torch import nn
from torchvision import datasets, transforms, models
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import time

import numpy as np


class SmoothingWrapper:  # Not a Module
    def __init__(self, model, sigma, batch_size=64):
        self.model = model
        self.sigma = sigma
        self.batch_size = batch_size

        
    def predict(self, x, alpha, mc_size=None):
        """ Gets prediction counts for MC
        :param x: the input [channel x height x width]
        :param alpha: the failure probability
        :param mc_size: the number of Monte Carlo samples to use
        :return: predicted class or None
        """
        counts = self.get_prediction_counts(x, mc_size)
        count1, count2 = counts.sort(descending=True)[:2]
        if binom_test(count1, count1 + count2) < alpha:
            return counts.argmax().item()
        else:
            return None
        

    def get_prediction_counts(self, x, mc_size=None):
        """ Gets prediction counts for MC
        :param x: the input [channel x height x width]
        :param mc_size: the number of Monte Carlo samples to use
        :return: counts themselves
        """
        if mc_size is None:
            mc_size = self.batch_size
        counts = None
        with torch.no_grad():
            for processed in range(0, mc_size, self.batch_size):
                sz = min(self.batch_size, mc_size - processed)
                outputs = self.model(self._get_noised_inputs(x, sz))
                add = outputs.argmax(1).bincount(minlength=outputs.shape[1])
                if counts is None:
                    counts = add
                else:
                    counts += add
        return counts
            
        
    def _get_noised_inputs(self, x, size):
        """ Gets noised inputs
        :param x: the input [channel x height x width]
        :return:[size x ch x heights x width] - samples from N(x, diag(self.sigma**2))
        """
        # x: [ch x heights x width]
        # returns [size x ch x heights x width] with size examples shifted by noise
        
        with torch.no_grad():
            x = x.expand((size, *x.shape))
            return x + torch.randn_like(x, device=x.device) * self.sigma
        
        
    def certify(self, x, n_sel, n_est, alpha):
        """ Certify radius with probability 1 - alpha
        :param x: the input [channel x height x width]
        :param n_sel: the number of Monte Carlo samples to use for selection
        :param n_est: the number of Monte Carlo samples to use for estimation
        :param alpha: the error probability
        :return: (predicted class, certified radius) or (None, None)
        """
        counts_selection = self.get_prediction_counts(x, n_sel)
        result_class = counts_selection.argmax().item()
        count_est = self.get_prediction_counts(x, n_est)[result_class].item()
        est_lo, est_hi = proportion_confint(count_est, n_est, 2 * alpha, method='beta')
        if est_lo < 0.05:
            return None, None
        else:
            return result_class, self.sigma * norm.ppf(est_lo)
        
        
    def eval(self):
        self.model.eval()
        
    def train(self):
        self.model.train()
        
    def get_training_output(self, x):
        """ Add noise to the sample for training and compute the model there
        :param x: the input sample [batch x channel x height x width]
        :return: output [batch x classes]
        """
        return self.model(x + torch.randn_like(x, device=x.device) * self.sigma)

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.resnet18 = models.resnet18(pretrained=True)
        self.resnet18.conv1 = nn.Conv2d(1, 64, 7, 2, 3, bias=False)
        self.resnet18.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet18(x)

class 

(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.resnet50 = models.resnet50(pretrained=True)
        self.resnet50.fc = nn.Linear(512, 10)

    def forward(self, x):
        return self.resnet50(x)


class PeriodicPrinter:
    def __init__(self, secs):
        self.secs = secs
        self.last_time = -secs

    def print(self, *args):
        cur_time = time.time()
        if self.last_time + self.secs < cur_time:
            print(*args)
            self.last_time = cur_time
            return True
        else:
            return False

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

    def zero(self, right_now=True):
        if right_now:
            self.total = self.value = 0


def train(loader, wrapper, criterion, optimizer, device=torch.device('cpu')):
    wrapper.train()
    loss_cnt = AverageCounter()
    printer = PeriodicPrinter(5)

    for i, (X, y) in enumerate(loader):
        loss = criterion(wrapper.get_training_output(X.to(device)), y.to(device))
        loss_cnt.add(loss.item())
        loss_cnt.zero(printer.print('{:.4f}'.format(loss_cnt.get_average())))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')


def test(dataset, wrapper, n, alpha, radii, device):
    wrapper.eval()
    radii = np.asarray(radii)
    accuracies = np.zeros(len(radii))
    printer = PeriodicPrinter(5)
    res = np.zeros(len(dataset))
    for i, (X, y) in enumerate(dataset):
        cl, radius = wrapper.certify(X.to(device), n, n, alpha)
        if cl == y:
            res[i] = radius
            accuracies[radii < radius] += 1
        else:
            res[i] = float('nan')
        if printer.print(f'Iteration {i + 1}'):
            for r, a in zip(radii, accuracies / (1 + i) * 100):
                 print(f'Accuracy @ {r} = {a}%')
    for r, a in zip(radii, accuracies / len(dataset)):
        print(f'Accuracy @ {r} = {a * 100}%')
    return res


def do_train(dataset):
    if dataset == 'mnist':
        train_dataset = datasets.MNIST('mnist/', download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
        test_dataset = datasets.MNIST('mnist/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
        model = MnistModel()
        sigma=1
    elif dataset == 'cifar':
        transform=transforms.Compose([transforms.ToTensor(),
             transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

        train_dataset = torchvision.datasets.CIFAR10(root='cifar/', train=True,
                                                download=True, transform=transform)

        test_dataset = torchvision.datasets.CIFAR10(root='cifar/', train=False,
                                               download=True, transform=transform)
        sigma=1


        model = CifarModel()
    else:
        raise ValueError()

    train_loader = torch.utils.data.DataLoader(train_dataset, 256)
    criterion = nn.CrossEntropyLoss()
    wrapper = SmoothingWrapper(model.to(device), sigma, 256)
    optimizer = torch.optim.Adam(wrapper.model.parameters(), lr=1e-3)
    epoch = 0
    while epoch < 100:
        epoch += 1
        print(f'Epoch {epoch}')
        train(train_loader, wrapper, criterion, optimizer, device)

    torch.save(dataset + '.pth')


def do_test(dataset):
    if dataset == 'mnist':
        test_dataset = datasets.MNIST('mnist/', train=False, download=True, transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ]))
        model = MnistModel()
        sigma=1
    elif dataset == 'cifar':
            test_dataset = torchvision.datasets.CIFAR10(root='cifar/', train=False,
                                                   download=True, transform=transform)
            sigma=1


        train_loader = torch.utils.data.DataLoader(train_dataset, 256)
        model = CifarModel()
    else:
        raise ValueError()

    model = torch.load(dataset + '.pth')
    wrapper = SmoothingWrapper(model.to(device), sigma, 256)
    test(mnist_test_dataset, wrapper, 10000, 0.001, [0, 1., 2.5, 3., 3.5, 4., 4.5, 5., 10.], device)



def main():
    import sys

    if len(sys.argv < 2):
        raise ValueError()

    if sys.argv[1] == 'train':
        assert len(sys.argv) == 3:
        do_train(sys.argv[2])
    elif sys.argv[1] == 'test':
        do_test(sys.argv[2])



if __name__ == '__main__':
    main()