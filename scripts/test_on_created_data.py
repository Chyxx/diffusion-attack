import os
import time

from utils.dataset import created_data_loader
from utils.func_util import clp, avgL2
from config import opt
from models.classifier import Classifier
import torch
from torchvision.transforms import Normalize
from utils.func_util import pert_to_norm, norm_to_pert

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


@torch.no_grad()
def test_process(classifiers, loader):
    for c in classifiers:
        c.eval()
    device = torch.device("cuda")
    total_old_eq = [0 for i in range(len(classifiers))]
    total_new_eq = [0 for i in range(len(classifiers))]
    img_count = 0

    channels_sum, channels_squared_sum, n_batchs = 0, 0, 0
    t = time.perf_counter()
    print("start val...")
    for batch_num, (imgs, adv_imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        adv_imgs = adv_imgs.to(device)
        labels = labels.to(device)
        old_pred_arg = [c(imgs).argmax(dim=1) for c in classifiers]
        new_pred_arg = [c(adv_imgs).argmax(dim=1) for c in classifiers]
        old_eq = [torch.eq(p, labels).sum().item() for p in old_pred_arg]
        new_eq = [torch.eq(p, labels).sum().item() for p in new_pred_arg]
        for i in range(len(classifiers)):
            total_old_eq[i] += old_eq[i]
            total_new_eq[i] += new_eq[i]
        # old_acc = [eq / labels.size(0) for eq in old_eq]
        # new_acc = [eq / labels.size(0) for eq in new_eq]
        img_count += labels.size(0)
        # print("test_batch: {}, img_num: {}".format(batch_num, labels.size(0)))
        # for i in range(len(classifiers)):
            # print("classifier: {}, old_acc: {}, new_acc: {}".format(i, old_acc[i], new_acc[i]))
        perts = pert_to_norm(adv_imgs - imgs)
        # perts = norm_to_pert(perts)
        channels_sum += torch.mean(perts, dim=[0,2,3])
        channels_squared_sum += torch.mean(perts ** 2, dim=[0,2,3])
        n_batchs += 1
    
    mean = channels_sum / n_batchs
    std = (channels_squared_sum / n_batchs - mean**2) ** 0.5
    print("val end")
    print("data_mean: {}, data_std: {}".format(mean, std))
    for i in range(len(classifiers)):
        print("classifier: {}, avg_old_acc: {}, avg_new_acc: {}".format(i, total_old_eq[i] / img_count, total_new_eq[i] / img_count))


def test_on_created_data(classifiers):
    train_loader, val_loader = created_data_loader(root=opt.create_data_path, batch_size=1024)
    test_process(classifiers, train_loader)
    test_process(classifiers, val_loader)


if __name__ == "__main__":
    # wind = Visdom()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_on_created_data([
        Classifier("resnet34").cuda(),
        Classifier("vgg16").cuda(),
        Classifier("vgg19bn").cuda()
    ])
