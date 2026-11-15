"""Independent observation-conditioned Q-functions over ACT action chunks."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import torch
import torchvision
from torch import Tensor, nn
from torchvision.models._utils import IntermediateLayerGetter
from torchvision.ops.misc import FrozenBatchNorm2d


def _positive_integer(value: int, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ValueError(f"ACT chunk critic {name} must be a positive integer")
    return value


def _hidden_dimensions(values: Sequence[int]) -> tuple[int, ...]:
    result = tuple(values)
    if not result or any(
        isinstance(value, bool) or not isinstance(value, int) or value < 1
        for value in result
    ):
        raise ValueError("ACT chunk critic hidden dimensions must be positive")
    return result


def _mlp(input_dim: int, hidden_dims: tuple[int, ...], output_dim: int) -> nn.Sequential:
    layers: list[nn.Module] = []
    previous = input_dim
    for hidden in hidden_dims:
        layers.extend((nn.Linear(previous, hidden), nn.ReLU()))
        previous = hidden
    layers.append(nn.Linear(previous, output_dim))
    return nn.Sequential(*layers)


class _ACTObservationEncoder(nn.Module):
    def __init__(
        self,
        act_config: object,
        *,
        feature_dim: int,
        require_visual_initialization: bool,
    ) -> None:
        super().__init__()
        from lerobot.policies.act.configuration_act import ACTConfig
        from lerobot.utils.constants import OBS_ENV_STATE, OBS_STATE

        if not isinstance(act_config, ACTConfig):
            raise TypeError("ACT chunk critic requires an official ACTConfig")
        if not isinstance(require_visual_initialization, bool):
            raise TypeError("ACT visual initialization flag must be boolean")
        self.feature_dim = _positive_integer(feature_dim, "observation_feature_dim")
        self.image_keys = tuple(act_config.image_features)
        self.vector_shapes: dict[str, tuple[int, ...]] = {}
        if act_config.robot_state_feature is not None:
            self.vector_shapes[OBS_STATE] = tuple(act_config.robot_state_feature.shape)
        if act_config.env_state_feature is not None:
            self.vector_shapes[OBS_ENV_STATE] = tuple(act_config.env_state_feature.shape)
        if any(len(shape) != 1 or shape[0] < 1 for shape in self.vector_shapes.values()):
            raise ValueError("ACT chunk critic supports vector state features only")
        self.image_shapes = {
            name: tuple(feature.shape)
            for name, feature in act_config.image_features.items()
        }
        if any(
            len(shape) != 3 or shape[0] != 3 or any(value < 1 for value in shape)
            for shape in self.image_shapes.values()
        ):
            raise ValueError("ACT chunk critic images must have shape (3, H, W)")

        image_feature_dim = 0
        if self.image_keys:
            model_factory = getattr(torchvision.models, act_config.vision_backbone, None)
            if model_factory is None:
                raise ValueError("ACT chunk critic vision backbone is unavailable")
            backbone_model = model_factory(
                replace_stride_with_dilation=[
                    False,
                    False,
                    bool(act_config.replace_final_stride_with_dilation),
                ],
                weights=None,
                norm_layer=FrozenBatchNorm2d,
            )
            self.backbone = IntermediateLayerGetter(
                backbone_model,
                return_layers={"layer4": "feature_map"},
            )
            self.pool = nn.AdaptiveAvgPool2d((1, 1))
            image_feature_dim = int(backbone_model.fc.in_features) * len(self.image_keys)
        else:
            self.backbone = None
            self.pool = None
        vector_dim = sum(shape[0] for shape in self.vector_shapes.values())
        total_dim = image_feature_dim + vector_dim
        if total_dim < 1:
            raise ValueError("ACT chunk critic requires at least one observation feature")
        self.projection = nn.Sequential(
            nn.Linear(total_dim, self.feature_dim),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
        )
        initialized = not self.image_keys or not require_visual_initialization
        self.register_buffer(
            "visual_initialization_complete",
            torch.tensor(initialized, dtype=torch.bool),
        )

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return (*self.vector_shapes, *self.image_keys)

    def initialize_from_actor(self, actor: nn.Module) -> None:
        from lerobot.policies.act.modeling_act import ACTPolicy

        if not isinstance(actor, ACTPolicy):
            raise TypeError("ACT critic visual initialization requires ACTPolicy")
        if tuple(actor.config.image_features) != self.image_keys:
            raise ValueError("ACT actor and critic image keys disagree")
        if not self.image_keys:
            self.visual_initialization_complete.fill_(True)
            return
        assert self.backbone is not None
        source = actor.model.backbone.state_dict()
        if source.keys() != self.backbone.state_dict().keys():
            raise ValueError("ACT actor and critic visual backbone structures disagree")
        self.backbone.load_state_dict(source, strict=True)
        self.visual_initialization_complete.fill_(True)

    def forward(self, observations: Mapping[str, Tensor]) -> Tensor:
        if not isinstance(observations, Mapping) or set(observations) != set(
            self.observation_keys
        ):
            raise KeyError("ACT critic observation keys disagree with ACTConfig")
        if not bool(self.visual_initialization_complete):
            raise RuntimeError("ACT critic visual backbone must be initialized from ACT")
        parameter = next(self.parameters())
        batch_size: int | None = None

        def validate(value: Tensor, shape: tuple[int, ...], name: str) -> Tensor:
            nonlocal batch_size
            if (
                not isinstance(value, Tensor)
                or value.ndim != len(shape) + 1
                or tuple(value.shape[1:]) != shape
                or value.shape[0] < 1
                or not value.is_floating_point()
                or value.dtype != parameter.dtype
                or value.device != parameter.device
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(f"ACT critic observation {name!r} has an invalid tensor")
            if batch_size is None:
                batch_size = int(value.shape[0])
            elif value.shape[0] != batch_size:
                raise ValueError("ACT critic observations must share batch size")
            return value

        features: list[Tensor] = []
        for name, shape in self.vector_shapes.items():
            features.append(validate(observations[name], shape, name).flatten(1))
        if self.image_keys:
            assert self.backbone is not None and self.pool is not None
            images = [
                validate(observations[name], self.image_shapes[name], name)
                for name in self.image_keys
            ]
            encoded = [
                self.pool(self.backbone(image)["feature_map"]).flatten(1)
                for image in images
            ]
            features.append(torch.cat(encoded, dim=-1))
        return self.projection(torch.cat(features, dim=-1))


class _ACTChunkEncoder(nn.Module):
    def __init__(
        self,
        execution_horizon: int,
        action_dim: int,
        feature_dim: int,
    ) -> None:
        super().__init__()
        self.execution_horizon = _positive_integer(
            execution_horizon,
            "execution_horizon",
        )
        self.action_dim = _positive_integer(action_dim, "action_dim")
        self.feature_dim = _positive_integer(feature_dim, "action_feature_dim")
        self.projection = nn.Sequential(
            nn.Linear(
                self.execution_horizon * (self.action_dim + 1),
                self.feature_dim,
            ),
            nn.LayerNorm(self.feature_dim),
            nn.ReLU(),
        )

    def forward(self, action_chunks: Tensor, executed_mask: Tensor) -> Tensor:
        parameter = next(self.parameters())
        if (
            not isinstance(action_chunks, Tensor)
            or action_chunks.ndim != 3
            or tuple(action_chunks.shape[1:])
            != (self.execution_horizon, self.action_dim)
            or action_chunks.shape[0] < 1
            or not action_chunks.is_floating_point()
            or action_chunks.dtype != parameter.dtype
            or action_chunks.device != parameter.device
            or not bool(torch.isfinite(action_chunks).all())
        ):
            raise ValueError("ACT critic actions must be finite (B, T, A)")
        batch_size = action_chunks.shape[0]
        if (
            not isinstance(executed_mask, Tensor)
            or executed_mask.shape != (batch_size, self.execution_horizon)
            or executed_mask.dtype != torch.bool
            or executed_mask.device != parameter.device
        ):
            raise ValueError("ACT critic executed_mask must be boolean (B, T)")
        lengths = executed_mask.to(torch.long).sum(dim=1)
        expected = torch.arange(
            self.execution_horizon,
            device=parameter.device,
        ).unsqueeze(0) < lengths.unsqueeze(1)
        if bool((lengths < 1).any()) or not torch.equal(executed_mask, expected):
            raise ValueError("ACT critic executed_mask must be an exact non-empty prefix")
        masked_actions = action_chunks.masked_fill(~executed_mask.unsqueeze(-1), 0.0)
        action_and_mask = torch.cat(
            (masked_actions, executed_mask.unsqueeze(-1).to(masked_actions.dtype)),
            dim=-1,
        )
        return self.projection(action_and_mask.flatten(1))


class ACTChunkQFunction(nn.Module):
    """One Q-function over an actor-ready observation and executed chunk."""

    def __init__(
        self,
        act_config: object,
        *,
        observation_feature_dim: int = 256,
        action_feature_dim: int = 256,
        hidden_dims: Sequence[int] = (512, 256),
        require_visual_initialization: bool = True,
    ) -> None:
        super().__init__()
        action_feature = getattr(act_config, "action_feature", None)
        prediction_horizon = getattr(act_config, "chunk_size", None)
        execution_horizon = getattr(act_config, "n_action_steps", None)
        if action_feature is None or len(action_feature.shape) != 1:
            raise ValueError("ACT chunk critic requires one vector action feature")
        self.prediction_horizon = _positive_integer(
            prediction_horizon,
            "prediction_horizon",
        )
        self.execution_horizon = _positive_integer(
            execution_horizon,
            "execution_horizon",
        )
        if self.execution_horizon > self.prediction_horizon:
            raise ValueError(
                "ACT critic execution horizon cannot exceed prediction horizon"
            )
        self.action_dim = _positive_integer(action_feature.shape[0], "action_dim")
        self.observation_encoder = _ACTObservationEncoder(
            act_config,
            feature_dim=observation_feature_dim,
            require_visual_initialization=require_visual_initialization,
        )
        self.action_encoder = _ACTChunkEncoder(
            self.execution_horizon,
            self.action_dim,
            action_feature_dim,
        )
        hidden = _hidden_dimensions(hidden_dims)
        self.q_head = _mlp(
            observation_feature_dim + action_feature_dim,
            hidden,
            1,
        )

    @property
    def observation_keys(self) -> tuple[str, ...]:
        return self.observation_encoder.observation_keys

    def initialize_visual_backbone_from_actor(self, actor: nn.Module) -> None:
        self.observation_encoder.initialize_from_actor(actor)

    def forward(
        self,
        observations: Mapping[str, Tensor],
        action_chunks: Tensor,
        executed_mask: Tensor,
    ) -> Tensor:
        observation_features = self.observation_encoder(observations)
        action_features = self.action_encoder(action_chunks, executed_mask)
        if observation_features.shape[0] != action_features.shape[0]:
            raise ValueError("ACT critic observation/action batches disagree")
        result = self.q_head(torch.cat((observation_features, action_features), dim=-1))
        if not bool(torch.isfinite(result).all()):
            raise RuntimeError("ACT critic returned NaN or Inf")
        return result


class ACTTwinChunkCritic(nn.Module):
    """Two parameter-independent ACT chunk Q-functions."""

    def __init__(self, act_config: object, **q_function_kwargs: object) -> None:
        super().__init__()
        self.q1 = ACTChunkQFunction(act_config, **q_function_kwargs)
        self.q2 = ACTChunkQFunction(act_config, **q_function_kwargs)
        self.prediction_horizon = self.q1.prediction_horizon
        self.execution_horizon = self.q1.execution_horizon
        self.action_dim = self.q1.action_dim
        self.observation_keys = self.q1.observation_keys

    def initialize_visual_backbones_from_actor(self, actor: nn.Module) -> None:
        self.q1.initialize_visual_backbone_from_actor(actor)
        self.q2.initialize_visual_backbone_from_actor(actor)

    def forward(
        self,
        observations: Mapping[str, Tensor],
        action_chunks: Tensor,
        executed_mask: Tensor,
    ) -> tuple[Tensor, Tensor]:
        return (
            self.q1(observations, action_chunks, executed_mask),
            self.q2(observations, action_chunks, executed_mask),
        )


__all__ = ["ACTChunkQFunction", "ACTTwinChunkCritic"]
