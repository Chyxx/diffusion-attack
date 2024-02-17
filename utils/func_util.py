import torch
import torch.nn as nn
from torchvision.transforms import Normalize
from config import opt


def one_hot(x, class_count, device):
    return torch.eye(class_count, device=device)[x, :]


def clp(x):
    """Limit value between 0 and 1"""
    return torch.clamp(x, 0, 1)


def targeted_func(x, class_count, t):
    """目标攻击时的损失函数"""
    k = 0
    # label = one_hot(torch.full([x.size(0)], t, dtype=torch.int), class_count).type(torch.int)
    # tmp = torch.where(label == 1, float('-inf'), x)
    # tmp = torch.log(tmp.max(dim=1)[0]) - torch.log(x[:, t])
    # return torch.where(tmp > -k, tmp, -k)
    idx = list(range(class_count))
    idx.pop(t)
    idx = torch.tensor(idx).cuda()
    tmp = torch.log(x.index_select(1, idx).max(dim=1)[0]) - torch.log(x[:, t])
    return torch.clamp(tmp, min=-k)


def untargeted_func(x, labels):
    """非目标攻击时的损失函数"""
    k = opt.margin
    class_count = opt.class_count
    target = one_hot(labels, class_count, x.device)
    with torch.no_grad():
        f0 = x[target == 1]
    f1 = x.scatter(1, labels.unsqueeze(dim=1), 0).max(dim=1)[0]
    return torch.clamp(torch.log(f0 + 1e-20) - torch.log(f1 + 1e-20), min = -k)


def DAloss(predict, labels, perturbation):
    return untargeted_func(predict, labels).mean() + opt.gamma * nn.functional.mse_loss(perturbation, torch.zeros_like(perturbation))


def avgL2(perturbation):
    return torch.sqrt(torch.sum(perturbation**2, dim=-3)).mean(dim=[-2, -1])


def pert_to_norm(perts):
    return Normalize(mean=[6.499e-5, 4.56e-5, 7.3849e-5], std=[0.0158, 0.0198, 0.0117])(perts)


def norm_to_pert(norm_perts):
    return Normalize(mean=[6.499e-5 / 0.0158, 4.56e-5 / 0.0198, 7.3849e-5 / 0.0117], std=[1/0.0158, 1/0.0198, 1/0.0117])(norm_perts)