from typing import Dict, Optional

import hydra
from omegaconf import DictConfig
import torch
import torch.nn as nn
from torch.nn.functional import mse_loss


class ArnoldConcatEncoders(nn.Module):
    def __init__(
        self,
        rgb0: DictConfig,
        proprio: DictConfig,
        device: torch.device,
        depth0: Optional[DictConfig] = None,
        rgb1: Optional[DictConfig] = None,
        depth1: Optional[DictConfig] = None,
        rgb2: Optional[DictConfig] = None,
        depth2: Optional[DictConfig] = None,
        rgb3: Optional[DictConfig] = None,
        depth3: Optional[DictConfig] = None,
        tactile: Optional[DictConfig] = None,
        state_decoder: Optional[DictConfig] = None,
    ):
        super().__init__()
        
        self._latent_size = rgb0.visual_features
        if rgb1:
            self._latent_size += rgb1.visual_features
        if rgb2:
            self._latent_size += rgb2.visual_features
        if rgb3:
            self._latent_size += rgb3.visual_features

        if depth0:
            self._latent_size += depth0.visual_features
        if depth1:
            self._latent_size += depth1.visual_features
        if depth2:
            self._latent_size += depth2.visual_features
        if depth3:
            self._latent_size += depth3.visual_features

        if tactile:
            self._latent_size += tactile.visual_features
        visual_features = self._latent_size
        # super ugly, fix this clip ddp thing in a better way
        if "clip" in rgb0["_target_"]:
            self.rgb0_encoder = hydra.utils.instantiate(rgb0, device=device)
        else:
            self.rgb0_encoder = hydra.utils.instantiate(rgb0)
        
        self.depth0_encoder = hydra.utils.instantiate(depth0) if depth0 else None
        
        if "clip" in rgb1["_target_"]:
            self.rgb1_encoder = hydra.utils.instantiate(rgb1, device=device)
        else:
            self.rgb1_encoder = hydra.utils.instantiate(rgb1)

        self.depth1_encoder = hydra.utils.instantiate(depth1) if depth0 else None

        if "clip" in rgb2["_target_"]:
            self.rgb2_encoder = hydra.utils.instantiate(rgb2, device=device)
        else:
            self.rgb2_encoder = hydra.utils.instantiate(rgb2)

        self.depth2_encoder = hydra.utils.instantiate(depth2) if depth2 else None

        if "clip" in rgb3["_target_"]:
            self.rgb3_encoder = hydra.utils.instantiate(rgb3, device=device)
        else:
            self.rgb3_encoder = hydra.utils.instantiate(rgb3)

        self.depth3_encoder = hydra.utils.instantiate(depth3) if depth0 else None


        # self.rgb_gripper_encoder = hydra.utils.instantiate(rgb_gripper) if rgb_gripper else None
        # self.depth_gripper_encoder = hydra.utils.instantiate(depth_gripper) if depth_gripper else None
        self.tactile_encoder = hydra.utils.instantiate(tactile)
        self.proprio_encoder = hydra.utils.instantiate(proprio)
        if self.proprio_encoder:
            self._latent_size += self.proprio_encoder.out_features

        self.state_decoder = None
        if state_decoder:
            state_decoder.visual_features = visual_features
            state_decoder.n_state_obs = self.proprio_encoder.out_features
            self.state_decoder = hydra.utils.instantiate(state_decoder)

        self.current_visual_embedding = None
        self.current_state_obs = None

    @property
    def latent_size(self):
        return self._latent_size

    def forward(
        self, imgs: Dict[str, torch.Tensor], depth_imgs: Dict[str, torch.Tensor], state_obs: torch.Tensor
    ) -> torch.Tensor:
        rgb0 = imgs["rgb0"]
        rgb1 = imgs["rgb1"] if "rgb1" in imgs else None
        rgb2 = imgs["rgb1"] if "rgb1" in imgs else None
        rgb3 = imgs["rgb1"] if "rgb1" in imgs else None


        rgb_tactile = imgs["rgb_tactile"] if "rgb_tactile" in imgs else None

        depth0 = depth_imgs["depth0"] if "depth0" in depth_imgs else None
        depth1 = depth_imgs["depth0"] if "depth0" in depth_imgs else None
        depth2 = depth_imgs["depth0"] if "depth0" in depth_imgs else None
        depth3 = depth_imgs["depth0"] if "depth0" in depth_imgs else None

        # depth_gripper = depth_imgs["depth_gripper"] if "depth_gripper" in depth_imgs else None

        b, s, c, h, w = rgb0.shape
        rgb0 = rgb0.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 200, 200)
        # ------------ Vision Network ------------ #
        encoded_imgs = self.rgb0_encoder(rgb0)  # (batch*seq_len, 64)
        encoded_imgs = encoded_imgs.reshape(b, s, -1)  # (batch, seq, 64)

        if depth0 is not None:
            depth0 = torch.unsqueeze(depth0, 2)
            depth0 = depth0.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 3, 200, 200)
            encoded_depth_0 = self.depth0_encoder(depth0)  # (batch*seq_len, 64)
            encoded_depth_0 = encoded_depth_0.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_depth_0], dim=-1)

        if rgb1 is not None:
            b, s, c, h, w = rgb1.shape
            rgb1 = rgb1.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_imgs_rgb1 = self.rgb1_encoder(rgb1)  # (batch*seq_len, 64)
            encoded_imgs_rgb1 = encoded_imgs_rgb1.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_rgb1], dim=-1)
            if depth1 is not None:
                depth1 = torch.unsqueeze(depth1, 2)
                depth1 = depth1.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 84, 84)
                encoded_depth1 = self.depth1_encoder(depth1)
                encoded_depth1 = encoded_depth1.reshape(b, s, -1)  # (batch, seq, 64)
                encoded_imgs = torch.cat([encoded_imgs, encoded_depth1], dim=-1)

        if rgb2 is not None:
            b, s, c, h, w = rgb2.shape
            rgb2 = rgb2.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_imgs_rgb2 = self.rgb1_encoder(rgb2)  # (batch*seq_len, 64)
            encoded_imgs_rgb2 = encoded_imgs_rgb2.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_rgb2], dim=-1)
            if depth2 is not None:
                depth2 = torch.unsqueeze(depth2, 2)
                depth2 = depth2.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 84, 84)
                encoded_depth2 = self.depth2_encoder(depth2)
                encoded_depth2 = encoded_depth2.reshape(b, s, -1)  # (batch, seq, 64)
                encoded_imgs = torch.cat([encoded_imgs, encoded_depth2], dim=-1)

        if rgb3 is not None:
            b, s, c, h, w = rgb3.shape
            rgb3 = rgb3.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_imgs_rgb3 = self.rgb3_encoder(rgb3)  # (batch*seq_len, 64)
            encoded_imgs_rgb3 = encoded_imgs_rgb3.reshape(b, s, -1)  # (batch, seq, 64)
            encoded_imgs = torch.cat([encoded_imgs, encoded_imgs_rgb3], dim=-1)
            if depth3 is not None:
                depth3 = torch.unsqueeze(depth3, 2)
                depth3 = depth3.reshape(-1, 1, h, w)  # (batch_size * sequence_length, 1, 84, 84)
                encoded_depth3 = self.depth3_encoder(depth1)
                encoded_depth3 = encoded_depth3.reshape(b, s, -1)  # (batch, seq, 64)
                encoded_imgs = torch.cat([encoded_imgs, encoded_depth3], dim=-1)
        

        if rgb_tactile is not None:
            b, s, c, h, w = rgb_tactile.shape
            rgb_tactile = rgb_tactile.reshape(-1, c, h, w)  # (batch_size * sequence_length, 3, 84, 84)
            encoded_tactile = self.tactile_encoder(rgb_tactile)
            encoded_tactile = encoded_tactile.reshape(b, s, -1)
            encoded_imgs = torch.cat([encoded_imgs, encoded_tactile], dim=-1)

        self.current_visual_embedding = encoded_imgs
        self.current_state_obs = state_obs  # type: ignore
        if self.proprio_encoder:
            state_obs_out = self.proprio_encoder(state_obs)
            perceptual_emb = torch.cat([encoded_imgs, state_obs_out], dim=-1)
        else:
            perceptual_emb = encoded_imgs

        return perceptual_emb

    def state_reconstruction_loss(self):
        assert self.state_decoder is not None
        proprio_pred = self.state_decoder(self.current_visual_embedding)
        return mse_loss(self.current_state_obs, proprio_pred)
