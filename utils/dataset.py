import glob
import os

import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.datasets as datasets


def tiny_loader(root, batch_size, shuffle=False):
    transform_train = transforms.Compose(
        [transforms.Resize(64), transforms.RandomHorizontalFlip(), transforms.ToTensor()])
    transform_val = transforms.Compose([transforms.Resize(64), transforms.ToTensor()])
    # root = "/home/data/tiny-imagenet-200/train"
    # root = "/home/data/tiny-imagenet-200/val"
    train_set = datasets.ImageFolder(root=root + "/train", transform=transform_train)
    val_set = datasets.ImageFolder(root=root + "/val", transform=transform_val)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=shuffle, pin_memory=True, num_workers=4)
    return train_loader, val_loader


def created_data_loader(root, batch_size):
    train_set = CreatedDataset(root + "/train", 64)
    val_set = CreatedDataset(root + "/val", 64)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True, pin_memory=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_set, batch_size=batch_size, shuffle=False, pin_memory=True, num_workers=4)
    return train_loader, val_loader


class CreatedDataset(Dataset):
    def __init__(self, root, resize):
        super(CreatedDataset, self).__init__()
        self.root = root
        self.resize = resize
        self.imgs = []
        self.adv_imgs = []
        self.labels = []

        labels = glob.glob(self.root + "/*")

        for label_path in labels:
            label = int(os.path.split(label_path)[1])
            imgs = glob.glob(os.path.join(label_path, "images/*.png"))
            adv_imgs = glob.glob((os.path.join(label_path, "adv_imgs/*.png")))
            for i in range(len(imgs)):
                self.imgs.append(imgs[i])
                self.adv_imgs.append(adv_imgs[i])
                self.labels.append(label)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, idx):
        img, adv_img, label = self.imgs[idx], self.adv_imgs[idx], self.labels[idx]
        tf = transforms.Compose([
            lambda x: Image.open(x).convert('RGB'),
            transforms.Resize(self.resize),
            transforms.ToTensor(),
        ])
        img = tf(img)
        adv_img = tf(adv_img)
        label = torch.tensor(label)
        return img, adv_img, label
