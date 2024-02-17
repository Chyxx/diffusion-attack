import os
import time

from utils.dataset import created_data_loader
from utils.func_util import pert_to_norm
from config import opt
from models.unet import DiffusionUNet
from models.gaussian_diffusion import DiffusionD
from models.resample import UniformSampler
import torch
from torchvision.utils import save_image

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def get_loss(imgs, adv_imgs, labels, ddpm, schedule_sampler, DD, device):
    imgs = imgs.to(device)
    adv_imgs = adv_imgs.to(device)
    labels = labels.to(device)
    t, weights = schedule_sampler.sample(labels.shape[0], device)
    losses = DD.compute_losses(ddpm, pert_to_norm(adv_imgs - imgs), imgs, t, labels)
    return (losses * weights).mean()


def learning_rate_decay(epoch_num):
    return 0.00005 * (0.5 ** epoch_num)


def train_diffusion(ddpm):
    train_loader, val_loader = created_data_loader(root=opt.create_data_path, batch_size=64)
    optimizer = torch.optim.AdamW(ddpm.parameters(), lr=0.00005)
    DD = DiffusionD()
    schedule_sampler = UniformSampler(DD)
    device = torch.device("cuda")
    for epoch_num in range(10):
        ddpm.train()
        lr = learning_rate_decay(epoch_num)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

            tt = time.perf_counter()
        for batch_num, (imgs, adv_imgs, labels) in enumerate(train_loader):
            loss = get_loss(imgs, adv_imgs, labels, ddpm, schedule_sampler, DD, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if batch_num % 150 == 0:
                print("epoch: {}, batch: {}, loss: {}, time: {}".format(epoch_num, batch_num, loss, time.perf_counter() - tt))
                tt = time.perf_counter()
            if batch_num % 150 == 0:
                print("start val...")
                ddpm.eval()
                t = time.perf_counter()
                avg_loss = 0
                for batch_num, (imgs, adv_imgs, labels) in enumerate(val_loader):
                    with torch.no_grad():
                        loss = get_loss(imgs, adv_imgs, labels, ddpm, schedule_sampler, DD, device)
                        avg_loss += loss * imgs.size(0) / len(val_loader.dataset)
                ddpm.train()
                print("avg_loss: {}, val_time: {}".format(avg_loss, time.perf_counter() - t))
                print("val end")
        print("start save...")
        ddpm.save(optimizer, epoch_num, batch_num)
        print("save end")


if __name__ == "__main__":
    # wind = Visdom()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train_diffusion(DiffusionUNet)


