#/usr/bin/env python3

from collections import deque
import time
import torch
from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
import numpy as np
from tqdm import notebook as tqdm

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

        
def get_prediction_counts(model, x, mc_size, batch_size, sigma, xrange=range):
    counts = None
    with torch.no_grad():
        for processed in xrange(0, mc_size, batch_size):
            sz = min(batch_size, mc_size - processed)
            noised_inputs = x.expand(sz, *x.shape)
            noised_inputs = noised_inputs + torch.randn_like(noised_inputs, device=noised_inputs.device) * sigma
            outputs = (model(noised_inputs) > 0).to(torch.int64)
            if counts is None:
                counts = torch.zeros(2, outputs.shape[1], dtype=torch.int64, device=outputs.device)
            add = torch.sum(outputs, dim=0)
            counts[1] += add
            counts[0] += sz - add
    return counts


from scipy.stats import norm, binom_test
from statsmodels.stats.proportion import proportion_confint
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
        if binom_test(count_est, n_est,p=0.5) < alpha:
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

if __name__ == '__main__':
    raise NotImplemented('this module is not for launching')