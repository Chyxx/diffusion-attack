import torch

from utils.dataset import tiny_loader
from utils.func_util import norm_to_pert
from models.gaussian_diffusion import DiffusionD
from config import opt
from utils.func_util import clp, avgL2, DAloss
from visdom import Visdom

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def attack(model, classifier, mode=0, estimator=None):
    wind = Visdom()
    model.eval()
    classifier.eval()
    if estimator is not None:
        estimator.eval()
    device = torch.device("cuda")
    batch_size = 64
    loader, _loader = tiny_loader(root=opt.data_path, batch_size=batch_size, shuffle=True)
    DD = DiffusionD()

    def cond_fn(x, t, imgs=None, labels=None):
        if mode == 1:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                perts = norm_to_pert(x_in)
                new_pred = classifier(clp(perts + imgs))
                old_pred_arg = classifier(imgs).argmax(dim=1)
                f = DAloss(new_pred, old_pred_arg, perts) * x.size(0)
                res = -1 * torch.autograd.grad(f, x_in)[0] * opt.grad_scale * 30
            # print(res)
            return res
        elif mode == 2:
            with torch.enable_grad():
                x_in = x.detach().requires_grad_(True)
                perts = norm_to_pert(x_in)
                l2Loss = torch.nn.functional.mse_loss(perts, torch.zeros_like(x)) * x.size(0)
            res = -1 * (estimator(x, t, imgs, labels) * 1e-3 + torch.autograd.grad(l2Loss, x_in)[0] * opt.gamma) * opt.grad_scale * 30
            # return -1 * torch.zeros_like(x)
            # print(res)
            # with torch.enable_grad():
            #     x_in = x.detach().requires_grad_(True)
            #     new_pred = classifier(clp(x_in / opt.pert_scale + imgs))
            #     old_pred_arg = classifier(imgs).argmax(dim=1)
            #     f = DAloss(new_pred, old_pred_arg, x_in / opt.pert_scale, 10000) * x.size(0)
            #     res = -1 * torch.autograd.grad(f, x_in)[0] * opt.grad_scale * 100
            # print(res)
            return res

    total_eq = 0
    total_num = 0
    for batch_num, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        # labels = torch.zeros_like(labels)
        model_kwargs = {"imgs": imgs, "labels": labels}
        norm_perts = DD.ddim_sample_loop(
            model,
            (batch_size, 3, opt.img_size, opt.img_size),
            clip_denoised=False,
            model_kwargs=model_kwargs,
            cond_fn=cond_fn if mode != 0 else None,
            device=device,
        )
        perts = norm_to_pert(norm_perts)

        adv_imgs = clp(imgs + perts)
        wind.images(clp(adv_imgs - imgs + 0.5).to("cpu"), win="pert")
        wind.images(adv_imgs.to("cpu"), win="imgs+pert")
        wind.images(imgs.to("cpu"), win="imgs")
        old_pred_arg = classifier(imgs).argmax(dim=1)
        new_pred_arg = classifier(adv_imgs).argmax(dim=1)
        old_eq = torch.eq(old_pred_arg, labels).sum().item()
        new_eq = torch.eq(new_pred_arg, labels).sum().item()
        old_acc = old_eq / labels.size(0)
        new_acc = new_eq / labels.size(0)
        eq = torch.eq(old_pred_arg, new_pred_arg).sum().item()
        succ_rate = 1 - eq / labels.size(0)
        print("batch: {}, avgL2: {}, eq: {}, succ_rate: {}".format(batch_num, avgL2(adv_imgs - imgs).mean(), eq, succ_rate))
        print("old_acc {}, new_acc: {}".format(old_acc, new_acc))

    print("--------------")
    print("total_succ_rate: {}".format(1 - total_eq / total_num))



