#!/usr/bin/env python

# Copyright 2024 Tony Z. Zhao and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Dict, List, Optional, Union

from .types import FeatureType, NormalizationMode, PolicyFeature


@dataclass
class ACTConfig:
    """Configuration class for the Action Chunking Transformers policy.

    Simplified version of Lerobot's ACTConfig adapted for LIBERO environment.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 100
    n_action_steps: int = 100

    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.MEAN_STD,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    input_features: Dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: Dict[str, PolicyFeature] = field(default_factory=dict)

    device: str = "cuda"
    use_amp: bool = False

    # Architecture.
    # Vision backbone.
    vision_backbone: str = "resnet18"
    pretrained_backbone_weights: Optional[str] = "ResNet18_Weights.IMAGENET1K_V1"
    replace_final_stride_with_dilation: bool = False
    # Transformer layers.
    pre_norm: bool = False
    dim_model: int = 512
    n_heads: int = 8
    dim_feedforward: int = 3200
    feedforward_activation: str = "relu"
    n_encoder_layers: int = 4
    n_decoder_layers: int = 1
    # VAE.
    use_vae: bool = True
    latent_dim: int = 32
    n_vae_encoder_layers: int = 4

    # Inference.
    temporal_ensemble_coeff: Optional[float] = None

    # Training and loss computation.
    dropout: float = 0.1
    kl_weight: float = 10.0

    # Training preset
    optimizer_lr: float = 1e-5
    optimizer_weight_decay: float = 1e-4
    optimizer_lr_backbone: float = 1e-5

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if not self.vision_backbone.startswith("resnet"):
            raise ValueError(
                f"`vision_backbone` must be one of the ResNet variants. Got {self.vision_backbone}."
            )
        if self.temporal_ensemble_coeff is not None and self.n_action_steps > 1:
            raise NotImplementedError(
                "`n_action_steps` must be 1 when using temporal ensembling. This is "
                "because the policy needs to be queried every step to compute the ensembled action."
            )
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.n_obs_steps != 1:
            raise ValueError(
                f"Multiple observation steps not handled yet. Got `nobs_steps={self.n_obs_steps}`"
            )

    def validate_features(self) -> None:
        if not self.image_features and not self.env_state_feature:
            raise ValueError("You must provide at least one image or the environment state among the inputs.")

    @property
    def observation_delta_indices(self) -> None:
        return None

    @property
    def action_delta_indices(self) -> List[int]:
        return list(range(self.chunk_size))

    @property
    def reward_delta_indices(self) -> None:
        return None

    @property
    def image_features(self) -> List[str]:
        """Keys of the input features that are images (camera)."""
        image_keys = []
        for key, feature in self.input_features.items():
            if feature.type == FeatureType.VISUAL:
                image_keys.append(key)
        return image_keys

    @property
    def robot_state_feature(self) -> Optional[PolicyFeature]:
        """Input feature that corresponds to the robot state."""
        return self.input_features.get("observation.state")

    @property
    def env_state_feature(self) -> Optional[PolicyFeature]:
        """Input feature that corresponds to the environment state."""
        return self.input_features.get("observation.environment_state")

    @property
    def action_feature(self) -> PolicyFeature:
        """Output feature that corresponds to actions."""
        return self.output_features["action"]

    @classmethod
    def from_json_file(cls, config_file: Union[str, Path]) -> "ACTConfig":
        """Load config from a JSON file."""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "ACTConfig":
        """Create config from a dictionary, filtering only relevant keys."""
        # Convert features to PolicyFeature objects
        input_features = {}
        for key, feature_dict in config_dict.get("input_features", {}).items():
            input_features[key] = PolicyFeature(
                type=FeatureType(feature_dict["type"]),
                shape=tuple(feature_dict["shape"])
            )

        output_features = {}
        for key, feature_dict in config_dict.get("output_features", {}).items():
            output_features[key] = PolicyFeature(
                type=FeatureType(feature_dict["type"]),
                shape=tuple(feature_dict["shape"])
            )

        # Convert normalization mapping
        normalization_mapping = {}
        for key, mode in config_dict.get("normalization_mapping", {}).items():
            normalization_mapping[key] = NormalizationMode(mode)

        # Define the keys we actually want from the config
        valid_keys = {
            'n_obs_steps', 'chunk_size', 'n_action_steps', 'device', 'use_amp',
            'vision_backbone', 'pretrained_backbone_weights', 'replace_final_stride_with_dilation',
            'pre_norm', 'dim_model', 'n_heads', 'dim_feedforward', 'feedforward_activation',
            'n_encoder_layers', 'n_decoder_layers', 'use_vae', 'latent_dim', 'n_vae_encoder_layers',
            'temporal_ensemble_coeff', 'dropout', 'kl_weight', 'optimizer_lr', 'optimizer_weight_decay',
            'optimizer_lr_backbone'
        }
        
        # Filter config_dict to only include keys that our ACTConfig supports
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Create config with converted features and filtered parameters
        config = cls(
            input_features=input_features,
            output_features=output_features,
            normalization_mapping=normalization_mapping,
            **filtered_config
        )
        
        return config
