# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn
import dmc2gym
import numpy as np
from torch.nn import functional as F

from encoder import make_encoder
from decoder import make_decoder
from sac_ae import weight_init
from train import parse_args
import utils


args = parse_args()
args.domain_name = 'walker'
args.task_name = 'walk'
args.image_size = 84
args.seed = 1
args.agent = 'bisim'
args.encoder_type = 'pixel'
args.action_repeat = 2
args.img_source = 'video'
args.num_layers = 4
args.num_filters = 32
args.hidden_dim = 1024
args.resource_files = '/datasets01/kinetics/070618/400/train/driving_car/*.mp4'
args.total_frames = 5000
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class VAE(nn.Module):
    def __init__(self, obs_shape):
        super().__init__()
        self.encoder = make_encoder(
            encoder_type='pixel', 
            obs_shape=obs_shape, 
            feature_dim=100, 
            num_layers=4, 
            num_filters=32).to(device)
        
        self.decoder = make_decoder(
                'pixel', obs_shape, 50, 4, 32).to(device)
        self.decoder.apply(weight_init)

    def train(self, obs):
        h = self.encoder(obs)
        mu, log_var = h[:, :50], h[:, 50:]
        eps = torch.randn_like(mu)
        reparam = mu + torch.exp(log_var / 2) * eps
        rec_obs = torch.sigmoid(self.decoder(reparam))
        BCE = F.binary_cross_entropy(rec_obs, obs / 255, reduction='sum')
        KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
        loss = BCE + KLD
        return loss


env = dmc2gym.make(
    domain_name=args.domain_name,
    task_name=args.task_name,
    resource_files=args.resource_files,
    img_source=args.img_source,
    total_frames=10,
    seed=args.seed,
    visualize_reward=False,
    from_pixels=(args.encoder_type == 'pixel'),
    height=args.image_size,
    width=args.image_size,
    frame_skip=args.action_repeat
)
env = utils.FrameStack(env, k=args.frame_stack)
vae = VAE(env.observation_space.shape)
train_dataset = torch.load('train_dataset.pt')
optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
train_loader = torch.utils.data.DataLoader(train_dataset['obs'], batch_size=32, shuffle=True)

# training loop
for i in range(100):
    total_loss = []
    for obs_batch in train_loader:
        optimizer.zero_grad()
        loss = vae.train(obs_batch.to(device).float())
        loss.backward()
        optimizer.step()
        total_loss.append(loss.item())

    print(np.mean(total_loss), i)

dataset = torch.load('dataset.pt')
with torch.no_grad():
    embeddings = vae.encoder(torch.FloatTensor(dataset['obs']).to(device)).cpu().numpy()
torch.save(embeddings, 'vae_embeddings.pt')