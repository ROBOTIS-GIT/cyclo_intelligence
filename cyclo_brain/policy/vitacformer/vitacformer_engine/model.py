#!/usr/bin/env python3
#
# Copyright 2026 ROBOTIS CO., LTD.
# Copyright (c) Facebook, Inc. and its affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""SH5 ViTacFormer checkpoint loader and inference policy.

The checkpoint used by Cyclo is an official ViTacFormer DETR/CVAE model
adapted to the SH5 data contract.  It predates LeRobot's ``config.json`` /
``safetensors`` format, so reconstruct the exact module graph here and expose
the small policy surface consumed by :mod:`vitacformer_engine`.

The transformer/backbone structure follows RoboVerseOrg/ViTacFormer commit
``d94788272b5f5e18a80bf62e3e5d51e7d2581d77`` (Apache-2.0).  SH5-specific
dimensions are read from and strictly checked against ``train_config.json``.
"""

from __future__ import annotations

import copy
import json
import logging
import math
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Optional

import torch
import torch.nn.functional as F
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter

logger = logging.getLogger("vitacformer_engine")

VITACFORMER_ARCHITECTURE_VERSION = "vitacformer_sh5_v2_1_right_persistence"
VITACFORMER_TACTILE_BATCH_KEY = "observation.tactile.vitacformer"
_IMAGE_KEY = "observation.images.rgb.cam_left_head"
_STATE_KEY = "observation.state"
_LEFT_TACTILE_KEY = "observation.tactile.left"
_RIGHT_TACTILE_KEY = "observation.tactile.right"
_EXPECTED_SOURCE_COMMIT = "d94788272b5f5e18a80bf62e3e5d51e7d2581d77"

# Exact 54-D command-space bounds used by the SH5 v2.1 trainer.
# This decoder preserves the trained policy semantics; command publishing
# uses the repository's common runtime and its simulation/robot mode gate.
_JOINT_LOWER = torch.tensor(
    [-3.14, 0.0, -3.14, -2.9361, -3.14, -1.57, -1.8201]
    + [-3.14, -3.14, -3.14, -2.9361, -3.14, -1.57, -1.5804]
    + [-1.57, 0.0, -1.57, -1.57] + [-0.6, 0.0, 0.0, 0.0] * 4
    + [-1.57, -3.14, 0.0, 0.0] + [-0.6, 0.0, 0.0, 0.0] * 4,
    dtype=torch.float32,
)
_JOINT_UPPER = torch.tensor(
    [3.14, 3.14, 3.14, 1.0786, 3.14, 1.57, 1.5804]
    + [3.14, 0.0, 3.14, 1.0786, 3.14, 1.57, 1.8201]
    + [1.57, 3.14, 0.0, 0.0] + [0.6, 2.0, 1.57, 1.57] * 4
    + [1.57, 0.0, 1.57, 1.57] + [0.6, 2.0, 1.57, 1.57] * 4,
    dtype=torch.float32,
)
_RIGHT_TACTILE_RESIDUAL_INDICES = torch.cat((
    torch.arange(45, 90),
    torch.arange(135, 180),
))


def _bound_action(
    action_logits: Tensor,
    action_mean: Tensor,
    action_std: Tensor,
    *,
    beta: float = 1000.0,
) -> tuple[Tensor, Tensor]:
    """Apply the trainer's smooth, inward-margined SH5 action decoder."""
    mean = action_mean.to(device=action_logits.device, dtype=torch.float32)
    std = action_std.to(device=action_logits.device, dtype=torch.float32)
    margin = 1e-5
    lower = _JOINT_LOWER.to(action_logits.device) + margin
    upper = _JOINT_UPPER.to(action_logits.device) - margin
    raw = action_logits.float() * std + mean
    bounded = (
        lower
        + F.softplus(raw - lower, beta=beta)
        - F.softplus(raw - upper, beta=beta)
    )
    bounded = torch.maximum(lower, torch.minimum(upper, bounded))
    return (bounded - mean) / std, bounded


def _apply_warm_start_ramp(
    action_raw: Tensor,
    current_state: Tensor,
    *,
    rows: int = 16,
) -> Tensor:
    """Reproduce the audited 16-row arm-only warm-start trajectory."""
    margin = 1e-5
    lower = _JOINT_LOWER.to(action_raw.device) + margin
    upper = _JOINT_UPPER.to(action_raw.device) - margin
    safe_current = torch.maximum(
        lower,
        torch.minimum(upper, current_state.float()),
    )
    count = min(int(rows), int(action_raw.shape[1]))
    alpha = torch.linspace(
        0.0,
        1.0,
        count,
        device=action_raw.device,
        dtype=torch.float32,
    )[None, :, None]
    target_arms = action_raw[:, count - 1:count, :14]
    ramp_arms = safe_current[:, None, :14] * (1.0 - alpha) + target_arms * alpha
    ramp = torch.cat((ramp_arms, action_raw[:, :count, 14:]), dim=-1)
    return torch.cat((ramp, action_raw[:, count:]), dim=1)


def _build_tactile_persistence(
    tactile_history: Tensor,
    tactile_future_mean: Tensor,
    tactile_future_std: Tensor,
) -> tuple[Tensor, Tensor]:
    """Build v2.1 learned-left/right-persistence tactile conditioning."""
    current_pressure = tactile_history[:, -1, :90]
    persistence_row = torch.cat(
        (current_pressure, torch.zeros_like(current_pressure)),
        dim=-1,
    )
    persistence = persistence_row[:, None, :].expand(-1, 18, -1)
    persistence = (
        persistence - tactile_future_mean
    ) / tactile_future_std
    residual_scale = torch.ones(
        180,
        device=tactile_history.device,
        dtype=tactile_history.dtype,
    )
    residual_scale.index_fill_(
        0,
        _RIGHT_TACTILE_RESIDUAL_INDICES.to(tactile_history.device),
        0.0,
    )
    return persistence, residual_scale


class _Feature:
    def __init__(self, shape: tuple[int, ...]):
        self.shape = shape


class _PositionEmbeddingSine(nn.Module):
    def __init__(
        self,
        num_pos_feats: int,
        temperature: int = 10000,
        normalize: bool = True,
        scale: Optional[float] = None,
    ) -> None:
        super().__init__()
        self.num_pos_feats = num_pos_feats
        self.temperature = temperature
        self.normalize = normalize
        self.scale = 2 * math.pi if scale is None else scale

    def forward(self, tensor: Tensor) -> Tensor:
        not_mask = torch.ones_like(tensor[0, [0]])
        y_embed = not_mask.cumsum(1, dtype=torch.float32)
        x_embed = not_mask.cumsum(2, dtype=torch.float32)
        if self.normalize:
            eps = 1e-6
            y_embed = y_embed / (y_embed[:, -1:, :] + eps) * self.scale
            x_embed = x_embed / (x_embed[:, :, -1:] + eps) * self.scale

        dim_t = torch.arange(
            self.num_pos_feats,
            dtype=torch.float32,
            device=tensor.device,
        )
        dim_t = self.temperature ** (
            2 * torch.div(dim_t, 2, rounding_mode="floor")
            / self.num_pos_feats
        )
        pos_x = x_embed[:, :, :, None] / dim_t
        pos_y = y_embed[:, :, :, None] / dim_t
        pos_x = torch.stack(
            (pos_x[:, :, :, 0::2].sin(), pos_x[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        pos_y = torch.stack(
            (pos_y[:, :, :, 0::2].sin(), pos_y[:, :, :, 1::2].cos()),
            dim=4,
        ).flatten(3)
        return torch.cat((pos_y, pos_x), dim=3).permute(0, 3, 1, 2)


class _FrozenBatchNorm2d(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.register_buffer("weight", torch.ones(channels))
        self.register_buffer("bias", torch.zeros(channels))
        self.register_buffer("running_mean", torch.zeros(channels))
        self.register_buffer("running_var", torch.ones(channels))

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        state_dict.pop(prefix + "num_batches_tracked", None)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def forward(self, tensor: Tensor) -> Tensor:
        weight = self.weight.reshape(1, -1, 1, 1)
        bias = self.bias.reshape(1, -1, 1, 1)
        running_var = self.running_var.reshape(1, -1, 1, 1)
        running_mean = self.running_mean.reshape(1, -1, 1, 1)
        scale = weight * (running_var + 1e-5).rsqrt()
        return tensor * scale + (bias - running_mean * scale)


class _BackboneBase(nn.Module):
    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.body = IntermediateLayerGetter(
            backbone,
            return_layers={"layer4": "0"},
        )
        self.num_channels = 512

    def forward(self, tensor: Tensor) -> OrderedDict[str, Tensor]:
        return self.body(tensor)


class _Backbone(_BackboneBase):
    def __init__(self) -> None:
        # Every backbone parameter is present in the checkpoint.  Avoid a
        # network-dependent ImageNet download during policy load.
        backbone = torchvision.models.resnet18(
            weights=None,
            replace_stride_with_dilation=[False, False, False],
            norm_layer=_FrozenBatchNorm2d,
        )
        super().__init__(backbone)


class _Joiner(nn.Sequential):
    def __init__(self, backbone: nn.Module, position_embedding: nn.Module) -> None:
        super().__init__(backbone, position_embedding)
        self.num_channels = 512

    def forward(self, tensor: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        features = self[0](tensor)
        outputs: list[Tensor] = []
        positions: list[Tensor] = []
        for feature in features.values():
            outputs.append(feature)
            positions.append(self[1](feature).to(feature.dtype))
        return outputs, positions


def _clones(module: nn.Module, count: int) -> nn.ModuleList:
    return nn.ModuleList([copy.deepcopy(module) for _ in range(count)])


class _TransformerEncoder(nn.Module):
    def __init__(
        self,
        encoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
    ) -> None:
        super().__init__()
        self.layers = _clones(encoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        pred_action: bool = False,
    ) -> Tensor:
        output = src
        for layer in self.layers:
            output = layer(
                output,
                src_mask=mask,
                src_key_padding_mask=src_key_padding_mask,
                pos=pos,
                pred_action=pred_action,
            )
        if self.norm is not None:
            output = self.norm(output)
        return output


class _TransformerDecoder(nn.Module):
    def __init__(
        self,
        decoder_layer: nn.Module,
        num_layers: int,
        norm: Optional[nn.Module] = None,
        return_intermediate: bool = False,
    ) -> None:
        super().__init__()
        self.layers = _clones(decoder_layer, num_layers)
        self.num_layers = num_layers
        self.norm = norm
        self.return_intermediate = return_intermediate

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        output = tgt
        intermediate = []
        for layer in self.layers:
            output = layer(
                output,
                memory,
                tgt_mask=tgt_mask,
                memory_mask=memory_mask,
                tgt_key_padding_mask=tgt_key_padding_mask,
                memory_key_padding_mask=memory_key_padding_mask,
                pos=pos,
                query_pos=query_pos,
            )
            if self.return_intermediate:
                intermediate.append(self.norm(output))
        if self.norm is not None:
            output = self.norm(output)
            if self.return_intermediate:
                intermediate.pop()
                intermediate.append(output)
        if self.return_intermediate:
            return torch.stack(intermediate)
        return output.unsqueeze(0)


class _TransformerEncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
        use_tactile: bool,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        if use_tactile:
            self.cross_attn_1 = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
            )
            self.cross_attn_2 = nn.MultiheadAttention(
                d_model,
                nhead,
                dropout=dropout,
            )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        if use_tactile:
            self.norm2 = nn.LayerNorm(d_model)
            self.norm3 = nn.LayerNorm(d_model)
        self.norm4 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        if use_tactile:
            self.dropout2 = nn.Dropout(dropout)
            self.dropout3 = nn.Dropout(dropout)
        self.dropout4 = nn.Dropout(dropout)
        self.activation = F.relu
        self.normalize_before = False
        self.use_tactile = use_tactile

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        src: Tensor,
        src_mask: Optional[Tensor] = None,
        src_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        pred_action: bool = False,
    ) -> Tensor:
        query = key = self.with_pos_embed(src, pos)
        update = self.self_attn(
            query,
            key,
            value=src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
        )[0]
        src = self.norm1(src + self.dropout1(update))

        if self.use_tactile:
            if pred_action:
                tactile_token, tactile_pos = src[:2], pos[:2]
                middle_tokens = src[2:4]
                other_tokens, other_pos = src[4:], pos[4:]
            else:
                tactile_token, tactile_pos = src[:1], pos[:1]
                middle_tokens = src[1:3]
                other_tokens, other_pos = src[3:], pos[3:]

            tactile_update = self.cross_attn_1(
                query=self.with_pos_embed(tactile_token, tactile_pos),
                key=self.with_pos_embed(other_tokens, other_pos),
                value=other_tokens,
            )[0]
            visual_update = self.cross_attn_2(
                query=self.with_pos_embed(other_tokens, other_pos),
                key=self.with_pos_embed(tactile_token, tactile_pos),
                value=tactile_token,
            )[0]
            tactile_token = self.norm2(
                tactile_token + self.dropout2(tactile_update)
            )
            other_tokens = self.norm3(
                other_tokens + self.dropout3(visual_update)
            )
            src = torch.cat(
                [tactile_token, middle_tokens, other_tokens],
                dim=0,
            )

        update = self.linear2(
            self.dropout(self.activation(self.linear1(src)))
        )
        return self.norm4(src + self.dropout4(update))


class _TransformerDecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.multihead_attn = nn.MultiheadAttention(
            d_model,
            nhead,
            dropout=dropout,
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)
        self.dropout3 = nn.Dropout(dropout)
        self.activation = F.relu
        self.normalize_before = False

    @staticmethod
    def with_pos_embed(tensor: Tensor, pos: Optional[Tensor]) -> Tensor:
        return tensor if pos is None else tensor + pos

    def forward(
        self,
        tgt: Tensor,
        memory: Tensor,
        tgt_mask: Optional[Tensor] = None,
        memory_mask: Optional[Tensor] = None,
        tgt_key_padding_mask: Optional[Tensor] = None,
        memory_key_padding_mask: Optional[Tensor] = None,
        pos: Optional[Tensor] = None,
        query_pos: Optional[Tensor] = None,
    ) -> Tensor:
        query = key = self.with_pos_embed(tgt, query_pos)
        update = self.self_attn(
            query,
            key,
            value=tgt,
            attn_mask=tgt_mask,
            key_padding_mask=tgt_key_padding_mask,
        )[0]
        tgt = self.norm1(tgt + self.dropout1(update))
        update = self.multihead_attn(
            query=self.with_pos_embed(tgt, query_pos),
            key=self.with_pos_embed(memory, pos),
            value=memory,
            attn_mask=memory_mask,
            key_padding_mask=memory_key_padding_mask,
        )[0]
        tgt = self.norm2(tgt + self.dropout2(update))
        update = self.linear2(
            self.dropout(self.activation(self.linear1(tgt)))
        )
        return self.norm3(tgt + self.dropout3(update))


class _Transformer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        d_model = 512
        encoder_layer = _TransformerEncoderLayer(
            d_model,
            nhead=8,
            dim_feedforward=3200,
            dropout=0.1,
            use_tactile=True,
        )
        self.encoder = _TransformerEncoder(encoder_layer, 4)
        decoder_layer = _TransformerDecoderLayer(
            d_model,
            nhead=8,
            dim_feedforward=3200,
            dropout=0.1,
        )
        self.decoder = _TransformerDecoder(
            decoder_layer,
            7,
            nn.LayerNorm(d_model),
            return_intermediate=True,
        )
        for parameter in self.parameters():
            if parameter.dim() > 1:
                nn.init.xavier_uniform_(parameter)
        self.d_model = d_model
        self.nhead = 8

    def forward(
        self,
        src: Tensor,
        mask: Optional[Tensor],
        query_embed: Tensor,
        pos_embed: Tensor,
        latent_input: Tensor,
        proprio_input: Tensor,
        additional_pos_embed: Tensor,
        tactile: Tensor,
        tactile_pred: Optional[Tensor],
    ) -> Tensor:
        batch_size = src.shape[0]
        src = src.flatten(2).permute(2, 0, 1)
        pos_embed = (
            pos_embed.flatten(2)
            .permute(2, 0, 1)
            .repeat(1, batch_size, 1)
        )
        query_embed = query_embed.unsqueeze(1).repeat(1, batch_size, 1)

        tokens = [tactile]
        positions = [additional_pos_embed[0]]
        if tactile_pred is not None:
            tokens.append(tactile_pred)
            positions.append(additional_pos_embed[1])
            pred_action = True
        else:
            pred_action = False
        tokens.extend([latent_input, proprio_input])
        positions.extend([additional_pos_embed[2], additional_pos_embed[3]])
        addition_input = torch.stack(tokens, dim=0)
        addition_pos = (
            torch.stack(positions, dim=0)
            .unsqueeze(1)
            .repeat(1, batch_size, 1)
        )
        src = torch.cat([addition_input, src], dim=0)
        pos_embed = torch.cat([addition_pos, pos_embed], dim=0)

        target = torch.zeros_like(query_embed)
        memory = self.encoder(
            src,
            src_key_padding_mask=mask,
            pos=pos_embed,
            pred_action=pred_action,
        )
        hidden = self.decoder(
            target,
            memory,
            memory_key_padding_mask=mask,
            pos=pos_embed,
            query_pos=query_embed,
        )
        return hidden.transpose(1, 2)


def _sinusoid_encoding_table(n_position: int, d_hid: int) -> Tensor:
    positions = torch.arange(n_position, dtype=torch.float32).unsqueeze(1)
    dimensions = torch.arange(d_hid, dtype=torch.float32).unsqueeze(0)
    angles = positions / torch.pow(
        10000,
        2 * torch.div(dimensions, 2, rounding_mode="floor") / d_hid,
    )
    table = torch.zeros(n_position, d_hid, dtype=torch.float32)
    table[:, 0::2] = torch.sin(angles[:, 0::2])
    table[:, 1::2] = torch.cos(angles[:, 1::2])
    return table.unsqueeze(0)


class _ViTacFormerCore(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        hidden_dim = 512
        self.num_queries = 100
        self.camera_names = [_IMAGE_KEY]
        self.transformer = _Transformer()
        action_encoder_layer = _TransformerEncoderLayer(
            hidden_dim,
            nhead=8,
            dim_feedforward=3200,
            dropout=0.1,
            use_tactile=False,
        )
        self.encoder = _TransformerEncoder(action_encoder_layer, 4)
        self.action_head = nn.Linear(hidden_dim, 54)
        self.is_pad_head = nn.Linear(hidden_dim, 1)
        self.query_embed = nn.Embedding(self.num_queries, hidden_dim)
        backbone = _Joiner(
            _Backbone(),
            _PositionEmbeddingSine(hidden_dim // 2),
        )
        self.input_proj = nn.Conv2d(
            backbone.num_channels,
            hidden_dim,
            kernel_size=1,
        )
        self.backbones = nn.ModuleList([backbone])
        self.input_proj_robot_state = nn.Linear(6 * 54, hidden_dim)
        self.input_proj_tactile = nn.Linear(18 * 180, hidden_dim)
        self.tactile_head = nn.Linear(hidden_dim, 180)
        self.query_embed_tactile = nn.Embedding(18, hidden_dim)
        self.latent_dim = 32
        self.cls_embed = nn.Embedding(1, hidden_dim)
        self.encoder_action_proj = nn.Linear(54, hidden_dim)
        self.encoder_joint_proj = nn.Linear(6 * 54, hidden_dim)
        self.latent_proj = nn.Linear(hidden_dim, self.latent_dim * 2)
        self.register_buffer(
            "pos_table",
            _sinusoid_encoding_table(102, hidden_dim),
        )
        self.latent_out_proj = nn.Linear(self.latent_dim, hidden_dim)
        self.additional_pos_embed = nn.Embedding(4, hidden_dim)

    def forward(
        self,
        state: Tensor,
        image: Tensor,
        tactile: Tensor,
        tactile_persistence: Tensor,
        tactile_residual_scale: Tensor,
    ) -> tuple[Tensor, Tensor]:
        batch_size = state.shape[0]
        latent_sample = torch.zeros(
            (batch_size, self.latent_dim),
            dtype=state.dtype,
            device=state.device,
        )
        latent_input = self.latent_out_proj(latent_sample)

        all_camera_features = []
        all_camera_positions = []
        for camera_index in range(image.shape[1]):
            features, positions = self.backbones[0](image[:, camera_index])
            all_camera_features.append(self.input_proj(features[0]))
            all_camera_positions.append(positions[0])
        src = torch.cat(all_camera_features, dim=3)
        pos = torch.cat(all_camera_positions, dim=3)
        proprio_input = self.input_proj_robot_state(state)
        tactile_input = self.input_proj_tactile(tactile)

        tactile_hidden = self.transformer(
            src,
            None,
            self.query_embed_tactile.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
            tactile_input,
            None,
        )[0]
        tactile_residual = self.tactile_head(tactile_hidden)
        if tuple(tactile_residual_scale.shape) != (180,):
            raise ValueError(
                "ViTacFormer tactile residual scale must be (180,), got "
                f"{tuple(tactile_residual_scale.shape)}"
            )
        tactile_residual_scale = tactile_residual_scale.to(
            dtype=tactile_residual.dtype,
        )
        tactile_hat = tactile_persistence + (
            tactile_residual * tactile_residual_scale
        )
        tactile_pred_input = self.input_proj_tactile(
            tactile_hat.reshape(batch_size, -1)
        )
        hidden = self.transformer(
            src,
            None,
            self.query_embed.weight,
            pos,
            latent_input,
            proprio_input,
            self.additional_pos_embed.weight,
            tactile_input,
            tactile_pred_input,
        )[0]
        return self.action_head(hidden), tactile_hat


class ViTacFormerPolicy(nn.Module):
    """Inference-only wrapper with SH5 normalization and temporal ensemble."""

    def __init__(self, train_config: dict[str, Any], stats: dict[str, Tensor]):
        super().__init__()
        contract = train_config["data_contract"]
        joint_names = [
            *[f"arm_l_joint{i}" for i in range(1, 8)],
            *[f"arm_r_joint{i}" for i in range(1, 8)],
            *[f"finger_l_joint{i}" for i in range(1, 21)],
            *[f"finger_r_joint{i}" for i in range(1, 21)],
        ]
        self.config = SimpleNamespace(
            type="vitacformer",
            architecture_version=VITACFORMER_ARCHITECTURE_VERSION,
            temporal_ensemble_coeff=0.01,
            input_features={
                _IMAGE_KEY: _Feature(
                    (
                        3,
                        int(contract["image_height"]),
                        int(contract["image_width"]),
                    )
                ),
                _STATE_KEY: _Feature((int(contract["state_dim"]),)),
                _LEFT_TACTILE_KEY: _Feature((45,)),
                _RIGHT_TACTILE_KEY: _Feature((45,)),
            },
            output_features={
                "action": _Feature((int(contract["action_dim"]),)),
            },
            observation_state_joint_names=joint_names,
            model_action_keys=[
                "arm_left",
                "arm_right",
                "hand_left",
                "hand_right",
            ],
            state_history_size=int(contract["state_history"]),
            state_history_hz=(
                float(contract["fps"]) / int(contract["state_stride"])
            ),
            tactile_history_size=int(contract["tactile_history"]),
            tactile_history_hz=float(contract["fps"]),
        )
        self.model = _ViTacFormerCore()
        for name in (
            "state_mean",
            "state_std",
            "action_mean",
            "action_std",
            "tactile_history_mean",
            "tactile_history_std",
            "tactile_future_mean",
            "tactile_future_std",
        ):
            self.register_buffer(
                f"_{name}",
                torch.as_tensor(stats[name], dtype=torch.float32).reshape(-1),
                persistent=False,
            )
        self.register_buffer(
            "_image_mean",
            torch.tensor([0.485, 0.456, 0.406]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_image_std",
            torch.tensor([0.229, 0.224, 0.225]).reshape(1, 1, 3, 1, 1),
            persistent=False,
        )
        self._temporal_chunks: list[tuple[int, Tensor]] = []

    def reset(self) -> None:
        self._temporal_chunks = []

    def predict_action_chunk(self, batch: dict[str, Tensor]) -> Tensor:
        state = batch[_STATE_KEY]
        tactile = batch[VITACFORMER_TACTILE_BATCH_KEY]
        image = batch[_IMAGE_KEY]
        if state.dim() != 3 or tuple(state.shape[1:]) != (6, 54):
            raise ValueError(
                "ViTacFormer state history must be (B, 6, 54), got "
                f"{tuple(state.shape)}"
            )
        if tactile.dim() != 3 or tuple(tactile.shape[1:]) != (18, 180):
            raise ValueError(
                "ViTacFormer tactile history must be (B, 18, 180), got "
                f"{tuple(tactile.shape)}"
            )
        if image.dim() != 4 or tuple(image.shape[1:]) != (3, 188, 336):
            raise ValueError(
                "ViTacFormer image must be (B, 3, 188, 336), got "
                f"{tuple(image.shape)}"
            )
        if not torch.isfinite(state).all() or not torch.isfinite(tactile).all():
            raise ValueError("ViTacFormer input contains NaN or Inf")

        normalized_state = (
            (state - self._state_mean) / self._state_std
        ).reshape(state.shape[0], -1)
        normalized_tactile = (
            (tactile - self._tactile_history_mean)
            / self._tactile_history_std
        ).reshape(tactile.shape[0], -1)
        tactile_persistence, tactile_residual_scale = (
            _build_tactile_persistence(
                tactile,
                self._tactile_future_mean,
                self._tactile_future_std,
            )
        )
        normalized_image = (
            image.unsqueeze(1) - self._image_mean
        ) / self._image_std
        normalized_action, _ = self.model(
            normalized_state,
            normalized_image,
            normalized_tactile,
            tactile_persistence,
            tactile_residual_scale,
        )
        _, action = _bound_action(
            normalized_action,
            self._action_mean,
            self._action_std,
        )
        action = _apply_warm_start_ramp(
            action,
            state[:, -1],
        )
        if not torch.isfinite(action).all():
            raise RuntimeError("ViTacFormer produced NaN or Inf actions")
        return action

    def select_action(self, batch: dict[str, Tensor]) -> Tensor:
        """Return the current absolute target from overlapping 100-step plans."""
        chunk = self.predict_action_chunk(batch)
        candidates: list[Tensor] = []
        ages: list[int] = []
        for age, previous in self._temporal_chunks:
            if age < previous.shape[1]:
                candidates.append(previous[:, age])
                ages.append(age)
        candidates.append(chunk[:, 0])
        ages.append(0)
        stacked = torch.stack(candidates, dim=0)
        weights = torch.exp(
            -float(self.config.temporal_ensemble_coeff)
            * torch.tensor(
                ages,
                dtype=stacked.dtype,
                device=stacked.device,
            )
        )
        weights = weights / weights.sum()
        selected = torch.sum(stacked * weights[:, None, None], dim=0)

        advanced = [
            (age + 1, previous)
            for age, previous in self._temporal_chunks
            if age + 1 < previous.shape[1]
        ]
        if chunk.shape[1] > 1:
            advanced.append((1, chunk.detach()))
        self._temporal_chunks = advanced
        return selected


def _resolve_vitacformer_layout(
    model_path: str | Path,
) -> tuple[Path, Optional[Path]] | None:
    """Resolve a run root, numeric checkpoint directory, or weight file.

    Cyclo's policy browser selects directories, while this artifact stores
    metadata at the run root and weights below ``checkpoints/``. A run root
    (or its ``checkpoints`` directory) means "best model"; selecting a
    numeric checkpoint directory means that directory's ``model.pt``.
    """
    raw_path = str(model_path or "").strip()
    if not raw_path:
        return None
    selected = Path(raw_path)

    explicit_weights: Optional[Path] = None
    if selected.is_file():
        if selected.suffix.lower() != ".pt":
            return None
        explicit_weights = selected
        start = selected.parent
    else:
        numeric_weights = selected / "model.pt"
        if numeric_weights.is_file() and not (selected / "train_config.json").is_file():
            explicit_weights = numeric_weights
        start = selected

    for root in (start, *list(start.parents)[:3]):
        config_path = root / "train_config.json"
        if not config_path.is_file():
            continue
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except (OSError, UnicodeError, json.JSONDecodeError):
            continue
        if "vitacformer" not in str(payload.get("architecture", "")).lower():
            continue
        if explicit_weights is not None:
            try:
                explicit_weights.relative_to(root)
            except ValueError:
                return None
        return root, explicit_weights
    return None


def is_vitacformer_model_dir(model_path: str | Path) -> bool:
    return _resolve_vitacformer_layout(model_path) is not None


def _validate_contract(config: dict[str, Any]) -> None:
    contract = config.get("data_contract")
    if not isinstance(contract, dict):
        raise ValueError("ViTacFormer train_config.json has no data_contract")
    expected = {
        "action_chunk": 100,
        "action_dim": 54,
        "state_dim": 54,
        "state_history": 6,
        "state_stride": 3,
        "tactile_history": 18,
        "tactile_raw_dim": 90,
        "tactile_rep_dim": 180,
        "tactile_taxels_per_hand": 45,
        "image_height": 188,
        "image_width": 336,
        "fps": 30,
        "image_key": _IMAGE_KEY,
    }
    # The original audited training export identifies the exact source commit
    # and all tensor dimensions, but predates these two descriptive keys.
    # Accept them when absent for that export; if a newer export includes them,
    # keep validating their values so contradictory preprocessing metadata
    # still fails closed.
    optional_annotations = {
        "tactile_order": "left[5,3,3],right[5,3,3]",
        "baseline": (
            "already corrected per episode from first 20 raw 100Hz samples"
        ),
    }
    mismatches = {
        key: (contract.get(key), value)
        for key, value in expected.items()
        if contract.get(key) != value
    }
    mismatches.update({
        key: (contract.get(key), value)
        for key, value in optional_annotations.items()
        if key in contract and contract.get(key) != value
    })
    if mismatches:
        raise ValueError(
            "Unsupported ViTacFormer SH5 data contract: "
            f"{mismatches}"
        )
    source_commit = str(config.get("source_repo_commit", ""))
    if source_commit != _EXPECTED_SOURCE_COMMIT:
        raise ValueError(
            "Unsupported ViTacFormer source commit: "
            f"{source_commit!r}; expected {_EXPECTED_SOURCE_COMMIT}"
        )


def _validate_stats(stats: Any) -> None:
    if not isinstance(stats, dict):
        raise TypeError("ViTacFormer normalization stats must be a dictionary")
    expected_shapes = {
        "state_mean": (54,),
        "state_std": (54,),
        "action_mean": (54,),
        "action_std": (54,),
        "tactile_history_mean": (180,),
        "tactile_history_std": (180,),
        "tactile_future_mean": (180,),
        "tactile_future_std": (180,),
    }
    for name, shape in expected_shapes.items():
        if name not in stats:
            raise ValueError(f"ViTacFormer stats are missing {name}")
        value = torch.as_tensor(stats[name])
        if tuple(value.shape) != shape:
            raise ValueError(
                f"ViTacFormer {name} must have shape {shape}, got "
                f"{tuple(value.shape)}"
            )
        if not torch.isfinite(value).all():
            raise ValueError(f"ViTacFormer {name} contains NaN or Inf")
        if name.endswith("_std") and not torch.all(value > 0):
            raise ValueError(f"ViTacFormer {name} must be strictly positive")


def _checkpoint_path(
    root: Path,
    explicit_weights: Optional[Path] = None,
) -> Path:
    if explicit_weights is not None:
        if not explicit_weights.is_file():
            raise FileNotFoundError(
                f"Selected ViTacFormer weights not found: {explicit_weights}"
            )
        return explicit_weights
    candidates = (
        root / "checkpoints" / "best_model.pt",
        root / "best_model.pt",
        root / "model.pt",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "ViTacFormer weights not found; expected checkpoints/best_model.pt, "
        "best_model.pt, or model.pt"
    )


def load_vitacformer_policy(
    model_path: str,
    device: torch.device,
) -> ViTacFormerPolicy:
    resolved = _resolve_vitacformer_layout(model_path)
    if resolved is None:
        raise ValueError(
            "Selected path is not a ViTacFormer run or checkpoint: "
            f"{model_path}"
        )
    root, explicit_weights = resolved
    config = json.loads((root / "train_config.json").read_text())
    _validate_contract(config)
    stats_path = root / "normalization_stats.pt"
    if not stats_path.is_file():
        raise FileNotFoundError(f"ViTacFormer stats not found: {stats_path}")
    stats = torch.load(stats_path, map_location="cpu", weights_only=True)
    _validate_stats(stats)
    policy = ViTacFormerPolicy(config, stats)

    weights_path = _checkpoint_path(root, explicit_weights)
    checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
    if not isinstance(checkpoint, dict):
        raise TypeError("ViTacFormer checkpoint must be a dictionary")
    state_dict = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    if not isinstance(state_dict, dict):
        raise TypeError("ViTacFormer checkpoint has no model state dictionary")
    policy.load_state_dict(state_dict, strict=True)
    policy.config.checkpoint_path = str(weights_path)
    policy = policy.to(device).eval()
    logger.info(
        "ViTacFormer SH5 weights loaded: %s (step=%s, device=%s)",
        weights_path,
        checkpoint.get("step", "unknown"),
        device,
    )
    return policy


__all__ = [
    "VITACFORMER_ARCHITECTURE_VERSION",
    "VITACFORMER_TACTILE_BATCH_KEY",
    "ViTacFormerPolicy",
    "is_vitacformer_model_dir",
    "load_vitacformer_policy",
]
