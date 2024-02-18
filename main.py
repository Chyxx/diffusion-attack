import os

from models.classifier import ClassifierToTrain, Classifier
from models.unet import DiffusionUNet, EstimatorUNet
from models.gaussian_diffusion import DiffusionD
from visdom import Visdom


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = input("Please choose a cuda:\n")
    cmd = int(input("To do?\n[0] train classifier\n[1] create_data\n[2] test_on_data\n[3] test_on_created_data\n"
                "[4] train_diffusion\n[5] train_estimator\n[6] attack\n"))
    if cmd == 0:
        from scripts.train_classifier import train_classifier
        train_classifier(ClassifierToTrain(input("Model?"), load=False).cuda())
    elif cmd == 1:
        from scripts.create_data import create_data
        create_data([
            Classifier("resnet34").cuda(),
            Classifier("vgg16").cuda(),
            Classifier("vgg19bn").cuda()
        ])
    elif cmd == 2:
        from scripts.test_on_data import test_on_data
        test_on_data([
            Classifier("resnet34").cuda(),
            Classifier("vgg16").cuda(),
            Classifier("vgg19bn").cuda()
        ])
    elif cmd == 3:
        from scripts.test_on_created_data import test_on_created_data
        test_on_created_data([
            Classifier("resnet34").cuda(),
            Classifier("vgg16").cuda(),
            Classifier("vgg19bn").cuda()
        ])
    elif cmd == 4:
        from scripts.train_diffusion import train_diffusion
        mode = int(input("Mode?\n[0] only on data\n[1] train by classifier grad\n"))
        if mode == 0:
            train_diffusion(DiffusionUNet().cuda(), mode)
        elif mode == 1:
            train_diffusion(DiffusionUNet().cuda(), mode, [
                Classifier("resnet34").cuda(),
                Classifier("vgg16").cuda(),
                Classifier("vgg19bn").cuda()
            ])
    elif cmd == 5:
        from scripts.train_estimator import train_estimator
        train_estimator(
            EstimatorUNet().cuda(),
            [
                Classifier("resnet34").cuda(),
                # Classifier("vgg16").cuda(),
                # Classifier("vgg19bn").cuda()
            ]
        )
    elif cmd == 6:
        from scripts.attack import attack
        mode = int(input("Mode?\n[0] without any guidance\n[1] guided by classifier\n[2] guided by estimator\n"))
        if mode == 0 or mode == 1:
            attack(DiffusionUNet(load=True).cuda(), Classifier("resnet34").cuda(), mode)
        else:
            attack(DiffusionUNet(load=True).cuda(), Classifier("resnet34").cuda(), mode, EstimatorUNet(load=True).cuda())


if __name__ == "__main__":
    main()

