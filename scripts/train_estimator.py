import os
import time

from utils.dataset import created_data_loader
from utils.func_util import pert_to_norm
from config import opt
from models.classifier import Classifier
from models.unet import EstimatorUNet
from models.gaussian_diffusion import EstimatorD
from models.resample import UniformSampler
import torch
from torchvision.utils import save_image
from visdom import Visdom

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def get_loss(imgs, adv_imgs, labels, estimator, classifiers, schedule_sampler, ED, device):
    imgs = imgs.to(device)
    adv_imgs = adv_imgs.to(device)
    labels = labels.to(device)
    t, weights = schedule_sampler.sample(labels.shape[0], device)
    losses = ED.compute_losses(estimator, classifiers, pert_to_norm(adv_imgs - imgs), imgs, t, labels)
    return (losses * weights).mean()


def train_estimator(estimator, classifiers):
    wind = Visdom()
    wind.line([0.], [0], win="training_loss(e)", opts=dict(title="training_loss_loss(e)"))
    for c in classifiers:
        c.eval()
    train_loader, val_loader = created_data_loader(root=opt.create_data_path, batch_size=128)
    optimizer = torch.optim.AdamW(estimator.parameters(), lr=1e-6)
    ED = EstimatorD()
    schedule_sampler = UniformSampler(ED)
    device = torch.device("cuda")
    steps = 0
    total_loss = 0
    loss_count = 0
    for epoch_num in range(1000):
        estimator.train()
        tt = time.perf_counter()
        for batch_num, (imgs, adv_imgs, labels) in enumerate(train_loader):
            loss = get_loss(imgs, adv_imgs, labels, estimator, classifiers, schedule_sampler, ED, device)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss
            loss_count += 1
            steps += 1
            if batch_num % 150 == 0:
                avg_loss = total_loss / loss_count
                print("epoch: {}, batch: {}, avg_loss: {}, time: {}".format(epoch_num, batch_num, avg_loss, time.perf_counter() - tt))
                wind.line([avg_loss.item()], [steps], update="append", opts=dict(title="training_loss(e)"), win="training_loss(e)")
                total_loss = 0
                loss_count = 0
                tt = time.perf_counter()
            # if batch_num % 150 == 0:
            #     print("start val...")
            #     estimator.eval()
            #     t = time.perf_counter()
            #     avg_loss = 0
            #     for batch_num, (imgs, adv_imgs, labels) in enumerate(val_loader):
            #         with torch.no_grad():
            #             loss = get_loss(imgs, adv_imgs, labels, estimator, classifiers, schedule_sampler, ED, device)
            #             avg_loss += loss * imgs.size(0) / len(val_loader.dataset)
            #             # print("val_batch: {}, loss: {}".format(batch_num, loss))
            #     estimator.train()
            #     print("avg_loss: {}, val_time: {}".format(avg_loss, time.perf_counter() - t))
            #     print("val end")
        print("start save...")
        estimator.save(optimizer, 0, 0)
        print("save end")

if __name__ == "__main__":
    # wind = Visdom()
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    train_estimator(
        EstimatorUNet().cuda(),
        [
            Classifier("resnet34").cuda(),
            Classifier("vgg16").cuda(),
            Classifier("vgg19bn").cuda()
        ]
    )

