import os

import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import transforms
from config import opt


class Classifier(nn.Module):
    def __init__(self, classifier_model, load=True):
        super(Classifier, self).__init__()
        if classifier_model == "resnet50":
            from torchvision.models import resnet50, ResNet50_Weights
            self.model = resnet50(weights=ResNet50_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, opt.class_count))
        elif classifier_model == "resnet34":
            from torchvision.models import resnet34, ResNet34_Weights
            self.model = resnet34(weights=ResNet34_Weights.DEFAULT)
            num_ftrs = self.model.fc.in_features
            self.model.fc = nn.Sequential(nn.Linear(num_ftrs, opt.class_count))
        elif classifier_model == "vgg19bn":
            from torchvision.models import vgg19_bn, VGG19_BN_Weights
            self.model = vgg19_bn(weights=VGG19_BN_Weights.DEFAULT)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, opt.class_count))
        elif classifier_model == "vgg16":
            from torchvision.models import vgg16, VGG16_Weights
            self.model = vgg16(weights=VGG16_Weights.DEFAULT)
            num_ftrs = self.model.classifier[-1].in_features
            self.model.classifier[-1] = nn.Sequential(nn.Linear(num_ftrs, opt.class_count))

        self.load_path = os.path.join(opt.classifier_path, classifier_model)
        self.save_path = os.path.join(opt.classifier_save_path, classifier_model)

        if load:
            checkpoint = torch.load(os.path.join(self.load_path, "_0_0_.pkl"))
            self.model.load_state_dict(checkpoint['model_state_dict'])

    def forward(self, x):
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).forward(x)
        return F.softmax(self.model(x), dim=1)

    def save(self, optimizer, epoch, batch):
        if not os.path.exists(self.save_path):
            os.makedirs(self.save_path)
        ckpt = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "batch": batch
        }
        torch.save(ckpt, os.path.join(self.save_path, "_{}_{}_.pkl".format(epoch, batch)))


class ClassifierToTrain(Classifier):
    def __init__(self, classifier_model, load=True):
        super().__init__(classifier_model, load)
    
    def forward(self, x):
        x = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]).forward(x)
        return self.model(x)
