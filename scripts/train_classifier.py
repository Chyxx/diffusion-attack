import os

import torch
from visdom import Visdom
from torch import nn
from torch.utils.data import DataLoader

from config import opt
from utils.dataset import tiny_loader
from utils.func_util import one_hot

seed = 23
torch.manual_seed(seed)  # 为CPU设置随机种子
torch.cuda.manual_seed(seed)  # 为当前GPU设置随机种子
torch.cuda.manual_seed_all(seed)  # 为所有GPU设置随机种子

wind = Visdom()

def train_classifier(classifier):
    total_train_step = 0
    val_min_loss = 1000000
    val_count = 0
    wind.line([0.], [0], win="val_loss", opts=dict(title="val_loss"))
    wind.line([0.], [0], win="val_acc", opts=dict(title="val_acc"))
    wind.line([0.], [0], win="train_loss", opts=dict(title="train_loss"))
    wind.line([0.], [0], win="train_acc", opts=dict(title="train_acc"))

    train_loader, val_loader = tiny_loader(root=opt.data_path, batch_size=64, shuffle=True)

    learning_rate = 0.001
    optimizer = torch.optim.SGD(classifier.parameters(), lr=learning_rate, momentum=0.9, weight_decay=1e-3)
    device = torch.device("cuda")
    criterion = nn.CrossEntropyLoss().to(device)
    total_train_step = 0
    for epoch_num in range(50):
        classifier.train()

        for batch_num, (imgs, labels) in enumerate(train_loader):
            imgs = imgs.to(device)
            labels = labels.to(device)
            pred = classifier(imgs)
            loss = criterion(pred, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_step = total_train_step + 1

            pred_arg = pred.argmax(dim=1)
            correct = torch.eq(pred_arg, labels).sum().item()

            if batch_num % 10 == 0:
                print("epoch：{}, batch: {}, loss: {}, acc: {}".format(epoch_num, batch_num, loss.item(), correct / labels.size(0)))
                wind.line([loss.item()], [total_train_step], update="append", opts=dict(title="train_loss"), win="train_loss")
                wind.line([correct / labels.size(0)], [total_train_step], update="append", opts=dict(title="train_acc"),
                          win="train_acc")


        classifier.eval()
        avg_loss = 0
        with torch.no_grad():
            correct = 0
            for batch_num, (imgs, labels) in enumerate(val_loader):
                imgs = imgs.to(device)
                labels = labels.to(device)

                pred = classifier(imgs)
                loss = criterion(pred, labels)
                avg_loss += loss.item() * labels.size(0) / len(val_loader.dataset)
                pred_arg = pred.argmax(dim=1)
                correct += pred_arg.eq(labels).sum().item()
            print("epoch：{}，avg_loss：{}, acc: {}".format(epoch_num, avg_loss,
                                                         correct / len(val_loader.dataset)))
            wind.line([avg_loss], [epoch_num], update="append", opts=dict(title="val_loss"), win="val_loss")
            wind.line([correct / len(val_loader.dataset)], [epoch_num], update="append", opts=dict(title="val_acc"),
                      win="val_acc")
        
        if avg_loss > val_min_loss:
            val_count += 1
            if val_count >= 3:
                break
        else:
            val_count = 0
            val_min_loss = avg_loss
            classifier.save(optimizer, 0, 0)

