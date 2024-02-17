import os
import time

from utils.dataset import tiny_loader
from utils.func_util import DAloss, clp, avgL2
from config import opt
from models.classifier import Classifier
import torch
from torchvision.utils import save_image
from visdom import Visdom

seed = 1
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子


def learning_rate_decay(i):
    lr = 1 * (0.95 ** (i // 10))
    return lr

wind = Visdom()

def process(classifiers, loader, root):
    img_count = 0
    for batch_num, (imgs, labels) in enumerate(loader):
        t = time.perf_counter()
        labels = labels.cuda()
        imgs = imgs.cuda()
        adv_imgs = torch.rand_like(imgs, requires_grad=True)
        optimizer = torch.optim.Adam([adv_imgs], lr=1)
        with torch.no_grad():
            old_pred_arg = [c(imgs).argmax(dim=1) for c in classifiers]
        for i in range(300):
            optimizer.zero_grad()
            new_pred = [c(adv_imgs) for c in classifiers]
            total_loss = 0
            for j in range(len(classifiers)):
                loss = DAloss(new_pred[j], old_pred_arg[j], adv_imgs - imgs) / len(classifiers)
                total_loss = total_loss + loss
            total_loss.backward()
            optimizer.step()
            with torch.no_grad():
                adv_imgs[:] = clp(adv_imgs)
            lr = learning_rate_decay(i)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr
            if i % 10 == 0:
                with torch.no_grad():
                    new_pred_arg = [p.argmax(dim=1) for p in new_pred]
                    eq = [torch.eq(old_pred_arg[j], new_pred_arg[j]).sum().float().item() for j in
                          range(len(classifiers))]
                    old_eq = [torch.eq(p, labels).sum().item() for p in old_pred_arg]
                    new_eq = [torch.eq(p, labels).sum().item() for p in new_pred_arg]
                    old_acc = [eq / labels.size(0) for eq in old_eq]
                    new_acc = [eq / labels.size(0) for eq in new_eq]
                    succ_rate = [1 - e / labels.size(0) for e in eq]
                print(
                    "batch_num: {}, i: {}, loss: {}, avgL2:{}".format(batch_num, i, total_loss,
                                                                      avgL2(adv_imgs - imgs).mean()))
                print("current_succ_rate: {} {} {}".format(succ_rate[0], succ_rate[1], succ_rate[2]))
                print("eq: {} {} {}".format(eq[0], eq[1], eq[2]))
                for i in range(len(classifiers)):
                    print("classifier: {}, old_acc: {}, new_acc: {}".format(i, old_acc[i], new_acc[i]))
                device = torch.device("cpu")
                wind.images(clp(adv_imgs - imgs + 0.5).to(device), win="pert")
                wind.images(adv_imgs.to(device), win="imgs+pert")
                wind.images(imgs.to(device), win="imgs")
        print("---------------\ntime: {}".format(time.perf_counter() - t))
        for i in range(labels.size(0)):
            path = root + "/{}".format(labels[i])
            if not os.path.exists(path + "/images"):
                os.makedirs(path + "/images", 0o0755)
            if not os.path.exists(path + "/adv_imgs"):
                os.makedirs(path + "/adv_imgs", 0o0755)
            save_image(imgs[i], path + "/images/{}.png".format(img_count))
            save_image(adv_imgs[i], path + "/adv_imgs/{}.png".format(img_count))
            img_count += 1
        


def create_data(classifiers):
    for c in classifiers:
        c.eval()
    train_loader, val_loader = tiny_loader(root=opt.data_path, batch_size=1024, shuffle=False)
    process(classifiers, train_loader, opt.create_data_path + "/train")
    process(classifiers, val_loader, opt.create_data_path + "/val")


if __name__ == "__main__":
    # wind = Visdom()
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    create_data([
        Classifier("resnet34").cuda(),
        Classifier("vgg16").cuda(),
        Classifier("vgg19bn").cuda()
    ])

