import os
import time

from utils.dataset import tiny_loader
from utils.func_util import clp, avgL2
from config import opt
from models.classifier import Classifier
import torch
from torchvision.utils import save_image

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


@torch.no_grad()
def test_process(classifiers, loader):
    for c in classifiers:
        c.eval()
    device = torch.device("cuda")
    total_eq = [0 for i in range(len(classifiers))]
    img_count = 0
    t = time.perf_counter()
    print("start val...")
    for batch_num, (imgs, labels) in enumerate(loader):
        imgs = imgs.to(device)
        labels = labels.to(device)
        pred_arg = [c(imgs).argmax(dim=1) for c in classifiers]
        eq = [torch.eq(p, labels).sum().item() for p in pred_arg]
        for i in range(len(classifiers)):
            total_eq[i] += eq[i]
        acc = [eq / labels.size(0) for eq in eq]
        img_count += labels.size(0)
        print("test_batch: {}, img_num: {}".format(batch_num, labels.size(0)))
        for i in range(len(classifiers)):
            print("classifier: {}, acc: {}".format(i, acc[i]))
    print("val end")
    for i in range(len(classifiers)):
        print("classifier: {}, avg_acc: {}".format(i, total_eq[i] / img_count))


def test_on_data(classifiers):
    train_loader, val_loader = tiny_loader(root=opt.data_path, batch_size=1024)
    test_process(classifiers, train_loader)
    test_process(classifiers, val_loader)


if __name__ == "__main__":
    # wind = Visdom()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    test_on_data([
        Classifier("resnet34").cuda(),
        Classifier("vgg16").cuda(),
        Classifier("vgg19bn").cuda()
    ])
