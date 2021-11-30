# Copyright (c) 2021 Rui Shu
import argparse
import numpy as np
import torch
import tqdm
from codebase import utils as ut
from codebase.models.vae import VAE
from codebase.train import train
from pprint import pprint
from torchvision import datasets, transforms
from VocalSetDataset import VocalSetDataset

parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--z',         type=int, default=4,    help="Number of latent dimensions")
parser.add_argument('--b',         type=float, default=0.01,    help="Number of latent dimensions")
parser.add_argument('--h',         type=int, default=64,    help="Number of hidden units per layer")
parser.add_argument('--iter_max',  type=int, default=30000, help="Number of training iterations")
parser.add_argument('--iter_save', type=int, default=10000, help="Save model every n iterations")
parser.add_argument('--run',       type=int, default=0,     help="Run ID. In case you want to run replicates")
parser.add_argument('--train',     type=int, default=1,     help="Flag for training")
parser.add_argument('--overwrite', type=int, default=0,     help="Flag for overwriting")
parser.add_argument('--data_dir', type=str, default='/Users/camillenoufi/cnoufi (not syncing)/Research/VQM-VAE/data/VocalSet/train/split_1s',     help="Full path to dataset directory containing measurements .xlsx file")
parser.add_argument('--meas_file', type=str, default='voicelab_results.xlsx',     help="filename of measurements .xlsx file")

args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# import ipdb; ipdb.set_trace()
print('Loading Dataset...')
VS = VocalSetDataset(args.data_dir, args.meas_file, device)
x_dim = list(VS[0][1].shape)[0]

# VS.df = VS._unnormalize_features()
train_loader, validation_loader = ut.partition_dataset(VS, batch_size=200, validation_split=0.05, shuffle_dataset=True)

layout = [
    ('model={:s}',  'vae'),
    ('z={:02d}',  args.z),
    ('b={:0.4f}', args.b),
    ('f={:02d}', x_dim),
    ('h={:04d}', args.h),
    ('run={:04d}', args.run),
]
model_name = '_'.join([t.format(v) for (t, v) in layout])
pprint(vars(args))
print('Model name:', model_name)

vae = VAE(z_dim=args.z, name=model_name, x_dim=x_dim, h_dim=args.h, beta=args.b).to(device)
print(vae)

print('Training Model...')
if args.train==1:
    writer = ut.prepare_writer(model_name, overwrite_existing=args.overwrite)
    train(model=vae,
          train_loader=train_loader,
          validation_loader=validation_loader,
          device=device,
          tqdm=tqdm.tqdm,
          writer=writer,
          iter_max=args.iter_max,
          iter_save=args.iter_save)
    # ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=args.train == 2)
else:
    ut.load_model_by_name(vae, global_step=args.iter_max, device=device)
    ut.evaluate_lower_bound(vae, labeled_subset, run_iwae=True)
