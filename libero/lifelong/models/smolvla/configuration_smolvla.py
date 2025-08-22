#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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
from typing import Dict, List, Optional, Union, Tuple

from .types import FeatureType, NormalizationMode, PolicyFeature


@dataclass
class SmolVLAConfig:
    """Configuration class for the SmolVLA policy.

    Simplified version of Lerobot's SmolVLAConfig adapted for LIBERO environment.
    """

    # Input / output structure.
    n_obs_steps: int = 1
    chunk_size: int = 50
    n_action_steps: int = 50

    normalization_mapping: Dict[str, NormalizationMode] = field(
        default_factory=lambda: {
            "VISUAL": NormalizationMode.IDENTITY,
            "STATE": NormalizationMode.MEAN_STD,
            "ACTION": NormalizationMode.MEAN_STD,
        }
    )

    input_features: Dict[str, PolicyFeature] = field(default_factory=dict)
    output_features: Dict[str, PolicyFeature] = field(default_factory=dict)

    device: str = "cuda"
    use_amp: bool = False

    # Shorter state and action vectors will be padded
    max_state_dim: int = 32
    max_action_dim: int = 32

    # Image preprocessing
    resize_imgs_with_padding: Tuple[int, int] = (512, 512)

    # Add empty images. Used by smolvla_aloha_sim which adds the empty
    # left and right wrist cameras in addition to the top camera.
    empty_cameras: int = 0

    # Converts the joint and gripper values from the standard Aloha space to
    # the space used by the pi internal runtime which was used to train the base model.
    adapt_to_pi_aloha: bool = False

    # Converts joint dimensions to deltas with respect to the current state before passing to the model.
    # Gripper dimensions will remain in absolute values.
    use_delta_joint_actions_aloha: bool = False

    # Tokenizer
    tokenizer_max_length: int = 48

    # Decoding
    num_steps: int = 10

    # Attention utils
    use_cache: bool = True

    # Finetuning settings
    freeze_vision_encoder: bool = True
    train_expert_only: bool = True
    train_state_proj: bool = True

    # Training presets
    optimizer_lr: float = 1e-4
    optimizer_betas: Tuple[float, float] = (0.9, 0.95)
    optimizer_eps: float = 1e-8
    optimizer_weight_decay: float = 1e-10
    optimizer_grad_clip_norm: float = 10

    scheduler_warmup_steps: int = 1_000
    scheduler_decay_steps: int = 30_000
    scheduler_decay_lr: float = 2.5e-6

    vlm_model_name: str = "HuggingFaceTB/SmolVLM2-500M-Video-Instruct"  # Select the VLM backbone.
    load_vlm_weights: bool = False  # Set to True in case of training the expert from scratch. True when init from pretrained SmolVLA weights

    add_image_special_tokens: bool = False  # Whether to use special image tokens around image features.

    attention_mode: str = "cross_attn"

    prefix_length: int = -1

    pad_language_to: str = "longest"  # "max_length"

    num_expert_layers: int = -1  # Less or equal to 0 is the default where the action expert has the same number of layers of VLM. Otherwise the expert have less layers.
    num_vlm_layers: int = 16  # Number of layers used in the VLM (first num_vlm_layers layers)
    self_attn_every_n_layers: int = 2  # Interleave SA layers each self_attn_every_n_layers
    expert_width_multiplier: float = 0.75  # The action expert hidden size (wrt to the VLM)

    min_period: float = 4e-3  # sensitivity range for the timestep used in sine-cosine positional encoding
    max_period: float = 4.0

    def __post_init__(self):
        """Input validation (not exhaustive)."""
        if self.n_action_steps > self.chunk_size:
            raise ValueError(
                f"The chunk size is the upper bound for the number of action steps per model invocation. Got "
                f"{self.n_action_steps} for `n_action_steps` and {self.chunk_size} for `chunk_size`."
            )
        if self.use_delta_joint_actions_aloha:
            raise NotImplementedError(
                "`use_delta_joint_actions_aloha` is used by smolvla for aloha real models. It is not ported yet in LeRobot."
            )

    def validate_features(self) -> None:
        for i in range(self.empty_cameras):
            key = f"observation.images.empty_camera_{i}"
            empty_camera = PolicyFeature(
                type=FeatureType.VISUAL,
                shape=(3, 480, 640),
            )
            self.input_features[key] = empty_camera

    @property
    def observation_delta_indices(self) -> List[int]:
        return [0]

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
    def from_json_file(cls, config_file: Union[str, Path]) -> "SmolVLAConfig":
        """Load config from a JSON file."""
        with open(config_file, 'r') as f:
            config_dict = json.load(f)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict) -> "SmolVLAConfig":
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
            'max_state_dim', 'max_action_dim', 'resize_imgs_with_padding', 'empty_cameras',
            'adapt_to_pi_aloha', 'use_delta_joint_actions_aloha', 'tokenizer_max_length',
            'num_steps', 'use_cache', 'freeze_vision_encoder', 'train_expert_only',
            'train_state_proj', 'optimizer_lr', 'optimizer_betas', 'optimizer_eps',
            'optimizer_weight_decay', 'optimizer_grad_clip_norm', 'scheduler_warmup_steps',
            'scheduler_decay_steps', 'scheduler_decay_lr', 'vlm_model_name', 'load_vlm_weights',
            'add_image_special_tokens', 'attention_mode', 'prefix_length', 'pad_language_to',
            'num_expert_layers', 'num_vlm_layers', 'self_attn_every_n_layers',
            'expert_width_multiplier', 'min_period', 'max_period'
        }
        
        # Filter config_dict to only include keys that our SmolVLAConfig supports
        filtered_config = {k: v for k, v in config_dict.items() if k in valid_keys}
        
        # Handle tuple conversion for certain fields
        if 'resize_imgs_with_padding' in filtered_config:
            filtered_config['resize_imgs_with_padding'] = tuple(filtered_config['resize_imgs_with_padding'])
        if 'optimizer_betas' in filtered_config:
            filtered_config['optimizer_betas'] = tuple(filtered_config['optimizer_betas'])
        
        # Create config with converted features and filtered parameters
        config = cls(
            input_features=input_features,
            output_features=output_features,
            normalization_mapping=normalization_mapping,
            **filtered_config
        )
        
        return config
