import argparse


parser = argparse.ArgumentParser()

# basic
parser.add_argument("--class_count", type=int, default=200, help="number of epochs of training")
parser.add_argument("--img_size", type=int, default=64, help="size of each image dimension")
parser.add_argument("--image_channels", type=int, default=3, help="number of image channels")

# unet
parser.add_argument("--model_channels", type=int, default=128)
parser.add_argument("--num_heads", type=int, default=1)
parser.add_argument("--num_head_channels", type=int, default=-1)
parser.add_argument("--num_res_blocks", type=int, default=3)
parser.add_argument("--resblock_updown", type=bool, default=True)
parser.add_argument("--use_new_attention_order", type=bool, default=True)
parser.add_argument("--use_scale_shift_norm", type=bool, default=True)
parser.add_argument("--use_fp16", type=bool, default=False)

# train
# parser.add_argument("--pert_scale", type=float, default=10.0)
parser.add_argument("--margin", type=int, default=50, help="margin in hinge_loss")
parser.add_argument("--gamma", type=float, default=10, help="gamma in DAloss")
parser.add_argument("--learn_sigma", type=bool, default=False)
parser.add_argument("--dropout", type=float, default=0.1)
parser.add_argument("--noise_schedule", type=str, default="cosine")
parser.add_argument("--use_kl", type=bool, default=False)

# sample
parser.add_argument("--timestep_respacing", type=str, default="ddim25")
parser.add_argument("--grad_scale", type=float, default=1.0)

# path
parser.add_argument("--data_path", default="/home/student/data/tiny-imagenet-200", type=str, help="path to data directory")
parser.add_argument("--create_data_path", default="data/created_data", type=str, help="path to data directory")
parser.add_argument("--classifier_path", type=str, default="data/classifier-ckpt")
parser.add_argument("--diffusion_path", type=str, default="data/diffusion-ckpt/_0_0_.pkl")
parser.add_argument("--estimator_path", type=str, default="data/estimator-ckpt/_0_0_.pkl")
parser.add_argument("--classifier_save_path", type=str, default="data/classifier-ckpt")
parser.add_argument("--diffusion_save_path", type=str, default="data/diffusion-ckpt")
parser.add_argument("--estimator_save_path", type=str, default="data/estimator-ckpt")

opt = parser.parse_args()
