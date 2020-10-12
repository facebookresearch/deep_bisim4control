# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
from sac_ae import  Actor, Critic, weight_init, LOG_FREQ
from transition_model import make_transition_model
from decoder import make_decoder


class BaselineAgent(object):
    """Baseline algorithm with transition model and various decoder types."""
    def __init__(
        self,
        obs_shape,
        action_shape,
        device,
        hidden_dim=256,
        discount=0.99,
        init_temperature=0.01,
        alpha_lr=1e-3,
        alpha_beta=0.9,
        actor_lr=1e-3,
        actor_beta=0.9,
        actor_log_std_min=-10,
        actor_log_std_max=2,
        actor_update_freq=2,
        critic_lr=1e-3,
        critic_beta=0.9,
        critic_tau=0.005,
        critic_target_update_freq=2,
        encoder_type='pixel',
        encoder_stride=2,
        encoder_feature_dim=50,
        encoder_lr=1e-3,
        encoder_tau=0.005,
        decoder_type='pixel',
        decoder_lr=1e-3,
        decoder_update_freq=1,
        decoder_weight_lambda=0.0,
        transition_model_type='deterministic',
        num_layers=4,
        num_filters=32
    ):
        self.device = device
        self.discount = discount
        self.critic_tau = critic_tau
        self.encoder_tau = encoder_tau
        self.actor_update_freq = actor_update_freq
        self.critic_target_update_freq = critic_target_update_freq
        self.decoder_update_freq = decoder_update_freq
        self.decoder_type = decoder_type
        self.hinge = 1.
        self.sigma = 0.5

        self.actor = Actor(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, actor_log_std_min, actor_log_std_max,
            num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target = Critic(
            obs_shape, action_shape, hidden_dim, encoder_type,
            encoder_feature_dim, num_layers, num_filters, encoder_stride
        ).to(device)

        self.critic_target.load_state_dict(self.critic.state_dict())

        self.transition_model = make_transition_model(
            transition_model_type, encoder_feature_dim, action_shape
        ).to(device)

        # optimizer for decoder
        self.decoder_optimizer = torch.optim.Adam(
            self.transition_model.parameters(),
            lr=decoder_lr,
            weight_decay=decoder_weight_lambda
        )

        # tie encoders between actor and critic
        self.actor.encoder.copy_conv_weights_from(self.critic.encoder)

        self.log_alpha = torch.tensor(np.log(init_temperature)).to(device)
        self.log_alpha.requires_grad = True
        # set target entropy to -|A|
        self.target_entropy = -np.prod(action_shape)

        self.decoder = None
        encoder_params = list(self.critic.encoder.parameters()) + list(self.transition_model.parameters())
        if decoder_type == 'pixel':
            # create decoder
            self.decoder = make_decoder(
                decoder_type, obs_shape, encoder_feature_dim, num_layers,
                num_filters
            ).to(device)
            self.decoder.apply(weight_init)
        elif decoder_type == 'inverse':
            self.inverse_model = nn.Sequential(
                nn.Linear(encoder_feature_dim * 2, 512),
                nn.LayerNorm(512),
                nn.ReLU(),
                nn.Linear(512, action_shape[0])).to(device)
            encoder_params += list(self.inverse_model.parameters())
        if decoder_type != 'identity':
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(encoder_params, lr=encoder_lr)
        if decoder_type == 'pixel':  # optimizer for decoder
            self.decoder_optimizer = torch.optim.Adam(
                self.decoder.parameters(),
                lr=decoder_lr,
                weight_decay=decoder_weight_lambda
            )
            # optimizer for critic encoder for reconstruction loss
            self.encoder_optimizer = torch.optim.Adam(
                self.critic.encoder.parameters(), lr=encoder_lr
            )

        # optimizers
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=actor_lr, betas=(actor_beta, 0.999)
        )

        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=critic_lr, betas=(critic_beta, 0.999)
        )

        self.log_alpha_optimizer = torch.optim.Adam(
            [self.log_alpha], lr=alpha_lr, betas=(alpha_beta, 0.999)
        )

        self.train()
        self.critic_target.train()

    def energy(self, state, action, next_state, no_trans=False):
        """Energy function based on normalized squared L2 norm."""

        norm = 0.5 / (self.sigma**2)

        if no_trans:
            diff = state - next_state
            normalization = 0.
        else:
            pred_trans_mu, pred_trans_sigma = self.transition_model(torch.cat([state, action], dim=1))
            if pred_trans_sigma is None:
                pred_trans_sigma = torch.Tensor([1.]).to(self.device)
            if isinstance(pred_trans_mu, list):  # i.e. comes from an ensemble
                raise NotImplementedError  # TODO: handle the additional ensemble dimension (0) in this case
            diff = (state + pred_trans_mu - next_state) / pred_trans_sigma
            normalization = torch.log(pred_trans_sigma)
        return norm * (diff.pow(2) + normalization).sum(1)

    def contrastive_loss(self, state, action, next_state):

        # Sample negative state across episodes at random
        batch_size = state.size(0)
        perm = np.random.permutation(batch_size)
        neg_state = state[perm]

        self.pos_loss = self.energy(state, action, next_state)
        zeros = torch.zeros_like(self.pos_loss)

        self.pos_loss = self.pos_loss.mean()
        self.neg_loss = torch.max(
            zeros, self.hinge - self.energy(
                state, action, neg_state, no_trans=True)).mean()

        loss = self.pos_loss + self.neg_loss

        return loss

    def train(self, training=True):
        self.training = training
        self.actor.train(training)
        self.critic.train(training)
        if self.decoder is not None:
            self.decoder.train(training)

    @property
    def alpha(self):
        return self.log_alpha.exp()

    def select_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, _, _, _ = self.actor(
                obs, compute_pi=False, compute_log_pi=False
            )
            return mu.cpu().data.numpy().flatten()

    def sample_action(self, obs):
        with torch.no_grad():
            obs = torch.FloatTensor(obs).to(self.device)
            obs = obs.unsqueeze(0)
            mu, pi, _, _ = self.actor(obs, compute_log_pi=False)
            return pi.cpu().data.numpy().flatten()

    def update_critic(self, obs, action, reward, next_obs, not_done, L, step):
        with torch.no_grad():
            _, policy_action, log_pi, _ = self.actor(next_obs)
            target_Q1, target_Q2 = self.critic_target(next_obs, policy_action)
            target_V = torch.min(target_Q1,
                                 target_Q2) - self.alpha.detach() * log_pi
            target_Q = reward + (not_done * self.discount * target_V)

        # get current Q estimates
        current_Q1, current_Q2 = self.critic(obs, action, detach_encoder=False)
        critic_loss = F.mse_loss(current_Q1,
                                 target_Q) + F.mse_loss(current_Q2, target_Q)
        L.log('train_critic/loss', critic_loss, step)

        # Optimize the critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        self.critic.log(L, step)

    def update_actor_and_alpha(self, obs, L, step):
        # detach encoder, so we don't update it with the actor loss
        _, pi, log_pi, log_std = self.actor(obs, detach_encoder=True)
        actor_Q1, actor_Q2 = self.critic(obs, pi, detach_encoder=True)

        actor_Q = torch.min(actor_Q1, actor_Q2)
        actor_loss = (self.alpha.detach() * log_pi - actor_Q).mean()

        L.log('train_actor/loss', actor_loss, step)
        L.log('train_actor/target_entropy', self.target_entropy, step)
        entropy = 0.5 * log_std.shape[1] * (1.0 + np.log(2 * np.pi)
                                            ) + log_std.sum(dim=-1)
        L.log('train_actor/entropy', entropy.mean(), step)

        # optimize the actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.actor.log(L, step)

        self.log_alpha_optimizer.zero_grad()
        alpha_loss = (self.alpha *
                      (-log_pi - self.target_entropy).detach()).mean()
        L.log('train_alpha/loss', alpha_loss, step)
        L.log('train_alpha/value', self.alpha, step)
        alpha_loss.backward()
        self.log_alpha_optimizer.step()

    def update_decoder(self, obs, action, target_obs, L, step):  #  uses transition model
        # image might be stacked, just grab the first 3 (rgb)!
        assert target_obs.dim() == 4
        target_obs = target_obs[:, :3, :, :]

        h = self.critic.encoder(obs)
        next_h = self.transition_model.sample_prediction(torch.cat([h, action], dim=1))
        if target_obs.dim() == 4:
            # preprocess images to be in [-0.5, 0.5] range
            target_obs = utils.preprocess_obs(target_obs)
        rec_obs = self.decoder(next_h)
        loss = F.mse_loss(target_obs, rec_obs)

        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()

        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/ae_loss', loss, step)

        self.decoder.log(L, step, log_freq=LOG_FREQ)

    def update_contrastive(self, obs, action, next_obs, L, step):
        latent = self.critic.encoder(obs)
        next_latent = self.critic.encoder(next_obs)
        loss = self.contrastive_loss(latent, action, next_latent)
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()
        L.log('train_ae/contrastive_loss', loss, step)

    def update_inverse(self, obs, action, next_obs, L, step):
        non_final_mask = torch.tensor(tuple(map(lambda s: not (s == 0).all(), next_obs)), device=self.device).long()  # hack
        latent = self.critic.encoder(obs[non_final_mask])
        next_latent = self.critic.encoder(next_obs[non_final_mask].to(self.device).float())
        # pred_next_latent = self.transition_model(torch.cat([latent, action], dim=1))
        # fpred_action = self.inverse_model(latent, pred_next_latent)
        pred_action = self.inverse_model(torch.cat([latent, next_latent], dim=1))
        loss = F.mse_loss(pred_action, action[non_final_mask]) # + F.mse_loss(fpred_action, action)
        self.encoder_optimizer.zero_grad()
        loss.backward()
        self.encoder_optimizer.step()
        L.log('train_ae/inverse_loss', loss, step)

    def update(self, replay_buffer, L, step):
        if self.decoder_type == 'inverse':
            obs, action, reward, next_obs, not_done, k_obs = replay_buffer.sample(k=True)
        else:
            obs, action, _, reward, next_obs, not_done = replay_buffer.sample()

        L.log('train/batch_reward', reward.mean(), step)

        self.update_critic(obs, action, reward, next_obs, not_done, L, step)

        if step % self.actor_update_freq == 0:
            self.update_actor_and_alpha(obs, L, step)

        if step % self.critic_target_update_freq == 0:
            utils.soft_update_params(
                self.critic.Q1, self.critic_target.Q1, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.Q2, self.critic_target.Q2, self.critic_tau
            )
            utils.soft_update_params(
                self.critic.encoder, self.critic_target.encoder,
                self.encoder_tau
            )

        if self.decoder is not None and step % self.decoder_update_freq == 0:  # decoder_type is pixel
            self.update_decoder(obs, action, next_obs, L, step)

        if self.decoder_type == 'contrastive':
            self.update_contrastive(obs, action, next_obs, L, step)
        elif self.decoder_type == 'inverse':
            self.update_inverse(obs, action, k_obs, L, step)

    def save(self, model_dir, step):
        torch.save(
            self.actor.state_dict(), '%s/actor_%s.pt' % (model_dir, step)
        )
        torch.save(
            self.critic.state_dict(), '%s/critic_%s.pt' % (model_dir, step)
        )
        if self.decoder is not None:
            torch.save(
                self.decoder.state_dict(),
                '%s/decoder_%s.pt' % (model_dir, step)
            )

    def load(self, model_dir, step):
        self.actor.load_state_dict(
            torch.load('%s/actor_%s.pt' % (model_dir, step))
        )
        self.critic.load_state_dict(
            torch.load('%s/critic_%s.pt' % (model_dir, step))
        )
        if self.decoder is not None:
            self.decoder.load_state_dict(
                torch.load('%s/decoder_%s.pt' % (model_dir, step))
            )
