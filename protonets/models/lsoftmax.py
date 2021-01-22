import math

import torch
from torch import nn
from torch.autograd import Variable

from scipy.special import binom

def log_softmax(input, dim=None, T=None):
    soft_max = softmax(input, dim=dim, Tem=T)
    return torch.log(soft_max)


def softmax(input, dim=None, Tem=None):
    exp_input = torch.exp(input/Tem)
    exp_sum = torch.sum(exp_input, dim=dim).view(-1,1)
    exp_sum_copy = exp_sum.unsqueeze(0).view(-1,1)
    exp_sum = torch.cat((exp_sum, exp_sum_copy), dim=1)
    result = exp_input/exp_sum
    return result

def CrossEntropy(input, target):
    p_t = -target
    q_log = torch.log(input)
    a, b = input.shape[0], input.shape[1]
    result = torch.zeros([a])
    for i in range(a):
        p_x = p_t[i].view(1, -1)
        q_x = q_log[i].view(1, -1)
        result[i] = (p_x * q_x).sum()
    return result.mean()

class DistillingLoss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(DistillingLoss, self).__init__()

    def forward(self, input, target):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        p_t = -target
        q_log = torch.log(input)
        a, b = input.shape[0], input.shape[1]
        result = torch.zeros([a], requires_grad=True)
        for i in range(a):
            p_x = p_t[i].view(1, -1)
            q_x = q_log[i].view(1, -1)
            result[i] = (p_x * q_x).sum()
        return torch.mean(result)

class MetricLoss(torch.nn.Module):
    # 不要忘记继承Module
    def __init__(self):
        super(DistillingLoss, self).__init__()

    def forward(self, input, target):
        """output和target都是1-D张量,换句话说,每个样例的返回是一个标量.
        """
        p_t = -target
        q_log = torch.log(input)
        a, b = input.shape[0], input.shape[1]
        result = torch.zeros([a])
        for i in range(a):
            p_x = p_t[i].view(1, -1)
            q_x = q_log[i].view(1, -1)
            result[i] = (p_x * q_x).sum()
        return torch.mean(result)


class LSoftmaxLinear(nn.Module):

    def __init__(self, input_dim, output_dim, margin):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.margin = margin

        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))

        self.divisor = math.pi / self.margin
        self.coeffs = binom(margin, range(0, margin + 1, 2))
        self.cos_exps = range(self.margin, -1, -2)
        self.sin_sq_exps = range(len(self.cos_exps))
        self.signs = [1]
        for i in range(1, len(self.sin_sq_exps)):
            self.signs.append(self.signs[-1] * -1)

    def reset_parameters(self):
        nn.init.kaiming_normal(self.weight.data.t())

    def find_k(self, cos):
        acos = cos.acos()
        k = (acos / self.divisor).floor().detach()
        return k

    def forward(self, input, target=None):
        if self.training:
            assert target is not None
            logit = input.matmul(self.weight)
            batch_size = logit.size(0)
            logit_target = logit[range(batch_size), target]
            weight_target_norm = self.weight[:, target].norm(p=2, dim=0)
            input_norm = input.norm(p=2, dim=1)
            # norm_target_prod: (batch_size,)
            norm_target_prod = weight_target_norm * input_norm
            # cos_target: (batch_size,)
            cos_target = logit_target / (norm_target_prod + 1e-10)
            sin_sq_target = 1 - cos_target**2

            num_ns = self.margin//2 + 1
            # coeffs, cos_powers, sin_sq_powers, signs: (num_ns,)
            coeffs = Variable(input.data.new(self.coeffs))
            cos_exps = Variable(input.data.new(self.cos_exps))
            sin_sq_exps = Variable(input.data.new(self.sin_sq_exps))
            signs = Variable(input.data.new(self.signs))

            cos_terms = cos_target.unsqueeze(1) ** cos_exps.unsqueeze(0)
            sin_sq_terms = (sin_sq_target.unsqueeze(1)
                            ** sin_sq_exps.unsqueeze(0))

            cosm_terms = (signs.unsqueeze(0) * coeffs.unsqueeze(0)
                          * cos_terms * sin_sq_terms)
            cosm = cosm_terms.sum(1)
            k = self.find_k(cos_target)

            ls_target = norm_target_prod * (((-1)**k * cosm) - 2*k)
            logit[range(batch_size), target] = ls_target
            return logit
        else:
            assert target is None
            return input.matmul(self.weight)

if __name__ == '__main__':
    input = torch.tensor([[0.8, 0.2], [0.4, 0.6]])
    target = torch.tensor([[0.9, 0.1], [0.8, 0.2]])
    ce = CrossEntropy(input, target)
    print(ce)