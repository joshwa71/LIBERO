#!/usr/bin/env python

# Copyright 2025 HuggingFace Inc. team. All rights reserved.
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

"""
SmolVLA Policy adapted for LIBERO environment.

Simplified version of Lerobot's SmolVLA that can be loaded and used
within LIBERO without external dependencies.
"""

import math
import os
import re
import json
from collections import deque
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn
import safetensors.torch
import numpy as np

from transformers import AutoProcessor, AutoConfig, AutoModel, AutoModelForImageTextToText

from .constants import ACTION, OBS_IMAGES
from .configuration_smolvla import SmolVLAConfig
from .normalize import Normalize, Unnormalize
from .types import FeatureType, PolicyFeature
from .smolvlm_with_expert import SmolVLMWithExpertModel

# Matches ".soNNN", optionally followed by "-something", up to the "_buffer_" marker
_VARIANT_RE = re.compile(r"\.so\d+(?:-[\w]+)?_buffer_")


def canonicalise(k: str) -> str:
    """
    Remove dataset-variant markers like '.so100-blue_' or '.so100_' from a
    normalisation-buffer key.
    """
    return _VARIANT_RE.sub(".buffer_", k)


def standardise_state_dict(
    checkpoint: Dict[str, torch.Tensor], ref_keys: set, *, verbose: bool = True
) -> Tuple[Dict[str, torch.Tensor], list]:
    """
    • Re-keys `checkpoint ` so that every entry matches the *reference* key set.
    • If several variant keys collapse to the same canonical name we keep the
      first one and log the collision.
    • Returns the new dict + a list of entries that could not be matched.
    """
    out, collisions, unmatched = {}, {}, []

    for k, v in checkpoint.items():
        canon = canonicalise(k)
        if canon in ref_keys:
            if canon in out:  # duplicate after collapsing
                collisions.setdefault(canon, []).append(k)
            else:
                out[canon] = v
        else:
            unmatched.append(k)

    if verbose:
        for canon, variants in collisions.items():
            print(f"[standardise_state_dict] '{canon}'  ←  {variants}")
        if unmatched:
            print(f"[standardise_state_dict] kept {len(unmatched)} unmatched keys")

    out.update({k: checkpoint[k] for k in unmatched})
    return out, unmatched


def rename_checkpoint_keys(checkpoint: Dict, rename_str: str):
    """
    Renames keys in a checkpoint dictionary based on the given rename string.

    Args:
        checkpoint (dict): The checkpoint dictionary.
        rename_str (str): A string specifying key mappings in the format "old1//new1,old2//new2".

    Returns:
        dict: The modified checkpoint with renamed keys.
    """
    rename_dict = dict(pair.split("//") for pair in rename_str.split(","))

    new_checkpoint = {}
    for k, v in checkpoint.items():
        for old_key, new_key in rename_dict.items():
            if old_key in k:
                k = k.replace(old_key, new_key)
        new_checkpoint[k] = v
    return new_checkpoint


def get_safe_dtype(dtype: torch.dtype, device_type: str) -> torch.dtype:
    """Get safe dtype for device type."""
    if device_type == "cpu":
        return torch.float32
    return dtype


def create_sinusoidal_pos_embedding(
    time: torch.Tensor, dimension: int, min_period: float, max_period: float, device="cpu"
) -> Tensor:
    """Computes sine-cosine positional embedding vectors for scalar positions."""
    if dimension % 2 != 0:
        raise ValueError(f"dimension ({dimension}) must be divisible by 2")

    if time.ndim != 1:
        raise ValueError("The time tensor is expected to be of shape `(batch_size, )`.")

    dtype = get_safe_dtype(torch.float64, device.type if hasattr(device, 'type') else device)
    fraction = torch.linspace(0.0, 1.0, dimension // 2, dtype=dtype, device=device)
    period = min_period * (max_period / min_period) ** fraction

    # Compute the outer product
    scaling_factor = 1.0 / period * 2 * math.pi
    sin_input = scaling_factor[None, :] * time[:, None]
    pos_emb = torch.cat([torch.sin(sin_input), torch.cos(sin_input)], dim=1)
    return pos_emb


def make_att_2d_masks(pad_masks, att_masks):
    """Create 2D attention masks from padding and attention masks."""
    if att_masks.ndim != 2:
        raise ValueError(att_masks.ndim)
    if pad_masks.ndim != 2:
        raise ValueError(pad_masks.ndim)

    cumsum = torch.cumsum(att_masks, dim=1)
    att_2d_masks = cumsum[:, None, :] <= cumsum[:, :, None]
    pad_2d_masks = pad_masks[:, None, :] * pad_masks[:, :, None]
    att_2d_masks = att_2d_masks & pad_2d_masks
    return att_2d_masks


def resize_with_pad(img, width, height, pad_value=-1):
    """Resize image with padding to maintain aspect ratio."""
    if img.ndim != 4:
        raise ValueError(f"(b,c,h,w) expected, but {img.shape}")

    cur_height, cur_width = img.shape[2:]

    ratio = max(cur_width / width, cur_height / height)
    resized_height = int(cur_height / ratio)
    resized_width = int(cur_width / ratio)
    resized_img = F.interpolate(
        img, size=(resized_height, resized_width), mode="bilinear", align_corners=False
    )

    pad_height = max(0, int(height - resized_height))
    pad_width = max(0, int(width - resized_width))

    # pad on left and top of image
    padded_img = F.pad(resized_img, (pad_width, 0, pad_height, 0), value=pad_value)
    return padded_img


def pad_vector(vector, new_dim):
    """Pad vector to new dimension."""
    if vector.shape[-1] == new_dim:
        return vector
    shape = list(vector.shape)
    current_dim = shape[-1]
    shape[-1] = new_dim
    new_vector = torch.zeros(*shape, dtype=vector.dtype, device=vector.device)
    new_vector[..., :current_dim] = vector
    return new_vector


def pad_tensor(tensor, max_len, pad_value=0):
    """Pad tensor along sequence dimension to max_len."""
    b, d = tensor.shape[:2]
    
    padded_tensor = torch.full(
        (b, max_len, *tensor.shape[2:]), pad_value, dtype=tensor.dtype, device=tensor.device
    )
    padded_tensor[:, :d] = tensor
    
    return padded_tensor


class SmolVLAPolicy(nn.Module):
    """
    SmolVLA Policy adapted for LIBERO environment.
    
    Simplified version of Lerobot's SmolVLAPolicy that can be loaded and used
    within LIBERO without external Lerobot dependencies.
    """

    def __init__(
        self,
        config: SmolVLAConfig,
        dataset_stats: Optional[Dict[str, Dict[str, Tensor]]] = None,
    ):
        """
        Args:
            config: Policy configuration class instance.
            dataset_stats: Dataset statistics to be used for normalization.
        """
        super().__init__()
        config.validate_features()
        self.config = config

        self.normalize_inputs = Normalize(config.input_features, config.normalization_mapping, dataset_stats)
        self.normalize_targets = Normalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )
        self.unnormalize_outputs = Unnormalize(
            config.output_features, config.normalization_mapping, dataset_stats
        )

        # Initialize language tokenizer
        self.language_tokenizer = AutoProcessor.from_pretrained(self.config.vlm_model_name).tokenizer

        self.model = VLAFlowMatching(config)
        self.reset()

    def reset(self):
        """This should be called whenever the environment is reset."""
        self._queues = {
            ACTION: deque(maxlen=self.config.n_action_steps),
        }

    @torch.no_grad()
    def select_action(self, batch: Dict[str, Tensor], noise: Tensor = None) -> Tensor:
        """Select a single action given environment observations."""
        self.eval()
        batch = self._prepare_batch(batch)
        
        # Action queue logic for n_action_steps > 1
        if len(self._queues[ACTION]) == 0:
            actions = self._get_action_chunk(batch, noise)
            self._queues[ACTION].extend(actions.transpose(0, 1)[: self.config.n_action_steps])

        return self._queues[ACTION].popleft()

    @torch.no_grad()
    def predict_action_chunk(self, batch: Dict[str, Tensor], noise: Tensor = None) -> Tensor:
        """Predict a chunk of actions given environment observations."""
        self.eval()
        batch = self._prepare_batch(batch)
        actions = self._get_action_chunk(batch, noise)
        return actions

    def _get_action_chunk(self, batch: Dict[str, Tensor], noise: Tensor = None) -> Tensor:
        """Get action chunk from the model."""
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)

        actions = self.model.sample_actions(images, img_masks, lang_tokens, lang_masks, state, noise=noise)

        # Unpad actions
        original_action_dim = self.config.action_feature.shape[0]
        actions = actions[:, :, :original_action_dim]

        actions = self.unnormalize_outputs({ACTION: actions})[ACTION]
        return actions

    def _prepare_batch(self, batch: Dict[str, Tensor]) -> Dict[str, Tensor]:
        """Prepare batch for model input."""
        batch = self.normalize_inputs(batch)
        return batch

    def prepare_images(self, batch):
        """Apply SmolVLA preprocessing to the images."""
        images = []
        img_masks = []
        present_img_keys = [key for key in self.config.image_features if key in batch]

        if len(present_img_keys) == 0:
            raise ValueError(
                f"All image features are missing from the batch. At least one expected. "
                f"(batch: {batch.keys()}) (image_features:{self.config.image_features})"
            )

        # Preprocess image features present in the batch
        for key in present_img_keys:
            img = batch[key][:, -1, :, :, :] if batch[key].ndim == 5 else batch[key]
            if self.config.resize_imgs_with_padding is not None:
                img = resize_with_pad(img, *self.config.resize_imgs_with_padding, pad_value=0)

            # Normalize from range [0,1] to [-1,1] as expected by siglip
            img = img * 2.0 - 1.0

            bsize = img.shape[0]
            device = img.device
            if f"{key}_padding_mask" in batch:
                mask = batch[f"{key}_padding_mask"].bool()
            else:
                mask = torch.ones(bsize, dtype=torch.bool, device=device)
            images.append(img)
            img_masks.append(mask)

        return images, img_masks

    def prepare_language(self, batch) -> Tuple[Tensor, Tensor]:
        """Tokenize the text input."""
        device = batch["observation.state"].device
        tasks = batch["task"]
        if isinstance(tasks, str):
            tasks = [tasks]

        if len(tasks) == 1:
            tasks = [tasks[0] for _ in range(batch["observation.state"].shape[0])]

        tasks = [task if task.endswith("\n") else f"{task}\n" for task in tasks]

        tokenized_prompt = self.language_tokenizer.__call__(
            tasks,
            padding=self.config.pad_language_to,
            padding_side="right",
            max_length=self.config.tokenizer_max_length,
            return_tensors="pt",
        )
        lang_tokens = tokenized_prompt["input_ids"].to(device=device)
        lang_masks = tokenized_prompt["attention_mask"].to(device=device, dtype=torch.bool)

        return lang_tokens, lang_masks

    def prepare_state(self, batch):
        """Pad state."""
        state = batch["observation.state"][:, -1, :] if batch["observation.state"].ndim > 2 else batch["observation.state"]
        state = pad_vector(state, self.config.max_state_dim)
        return state

    def prepare_action(self, batch):
        """Pad action."""
        actions = pad_vector(batch[ACTION], self.config.max_action_dim)
        return actions

    def forward(self, batch: Dict[str, Tensor], noise=None, time=None) -> Dict[str, Tensor]:
        """Do a full training forward pass to compute the loss."""
        batch = self.normalize_inputs(batch)
        batch = self.normalize_targets(batch)
        images, img_masks = self.prepare_images(batch)
        state = self.prepare_state(batch)
        lang_tokens, lang_masks = self.prepare_language(batch)
        actions = self.prepare_action(batch)
        
        loss_dict = {}
        losses = self.model.forward(images, img_masks, lang_tokens, lang_masks, state, actions, noise, time)
        loss_dict["losses_after_forward"] = losses.clone()

        # Remove padding
        losses = losses[:, :, : self.config.max_action_dim]
        loss_dict["losses_after_rm_padding"] = losses.clone()

        # For backward pass
        loss = losses.mean()
        loss_dict["loss"] = loss.item()
        return loss, loss_dict

    @classmethod
    def from_pretrained(cls, pretrained_path: Union[str, Path]) -> "SmolVLAPolicy":
        """Load a pretrained SmolVLA policy from a directory containing model files."""
        pretrained_path = Path(pretrained_path)
        
        # Load config
        config_file = pretrained_path / "config.json"
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
        
        config = SmolVLAConfig.from_json_file(config_file)
        
        # Create policy instance (without dataset_stats - they're in the state_dict)
        policy = cls(config, dataset_stats=None)
        
        # Load model weights
        model_file = pretrained_path / "model.safetensors"
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        
        # Load state dict from safetensors
        state_dict = safetensors.torch.load_file(str(model_file), device=config.device)
        
        # Standardize state dict but preserve normalization keys
        policy_keys = set(policy.state_dict().keys())
        norm_keys_in_state = {k: v for k, v in state_dict.items() if any(norm in k for norm in ["normalize_inputs", "normalize_targets", "unnormalize_outputs"])}
        
        # Standardize non-normalization keys
        non_norm_state = {k: v for k, v in state_dict.items() if not any(norm in k for norm in ["normalize_inputs", "normalize_targets", "unnormalize_outputs"])}
        standardized_state, unmatched = standardise_state_dict(non_norm_state, policy_keys)
        
        # Add back normalization keys without standardization
        state_dict = {**standardized_state, **norm_keys_in_state}
        
        # Keep normalization keys - they should be loaded from the trained model
        # norm_keys = ("normalize_inputs", "normalize_targets", "unnormalize_outputs")
        # state_dict = {k: v for k, v in state_dict.items() if not k.startswith(norm_keys)}
        
        # Load the state dict
        missing_keys, unexpected_keys = policy.load_state_dict(state_dict, strict=False)
        
        # Debug the loading
        print(f"Missing keys: {len(missing_keys)} keys")
        print(f"Unexpected keys: {len(unexpected_keys)} keys")
        
        # Check for normalization parameters specifically
        norm_missing = [key for key in missing_keys if any(norm in key for norm in ["normalize_inputs", "normalize_targets", "unnormalize_outputs"])]
        if norm_missing:
            print(f"WARNING: Missing normalization keys: {norm_missing[:5]}...")  # Show first 5
            
        # Also check what normalization keys were loaded
        norm_loaded = [key for key in state_dict.keys() if any(norm in key for norm in ["normalize_inputs", "normalize_targets", "unnormalize_outputs"])]
        if norm_loaded:
            print(f"Loaded normalization keys: {norm_loaded[:5]}...")  # Show first 5
        else:
            print("WARNING: No normalization keys found in state_dict!")
            
        # Check if we have any buffer keys
        buffer_keys = [key for key in state_dict.keys() if "buffer_" in key]
        if buffer_keys:
            print(f"Buffer keys found: {buffer_keys[:5]}...")  # Show first 5
        
        policy.to(config.device)
        policy.eval()
        
        # Debug normalization stats after loading
        for name, module in policy.named_modules():
            if hasattr(module, 'features') and 'normalize' in name:
                for feature_name in module.features:
                    buffer_name = "buffer_" + feature_name.replace(".", "_")
                    if hasattr(module, buffer_name):
                        buffer = getattr(module, buffer_name)
                        if hasattr(buffer, 'mean'):
                            mean_val = buffer.mean.data
                            print(f"{name}.{buffer_name}.mean: min={mean_val.min():.4f}, max={mean_val.max():.4f}, isinf={torch.isinf(mean_val).any()}")
                            if torch.isinf(mean_val).any():
                                print(f"  ERROR: Infinity values in {name}.{buffer_name}.mean!")
        
        return policy


class VLAFlowMatching(nn.Module):
    """
    SmolVLA Flow Matching model adapted for LIBERO.
    """

    def __init__(self, config: SmolVLAConfig):
        super().__init__()
        self.config = config

        # Initialize the VLM with expert model
        self.vlm_with_expert = SmolVLMWithExpertModel(
            model_id=self.config.vlm_model_name,
            freeze_vision_encoder=self.config.freeze_vision_encoder,
            train_expert_only=self.config.train_expert_only,
            load_vlm_weights=self.config.load_vlm_weights,
            attention_mode=self.config.attention_mode,
            num_expert_layers=self.config.num_expert_layers,
            num_vlm_layers=self.config.num_vlm_layers,
            self_attn_every_n_layers=self.config.self_attn_every_n_layers,
            expert_width_multiplier=self.config.expert_width_multiplier,
        )

        # State and action projections
        self.state_proj = nn.Linear(
            self.config.max_state_dim, self.vlm_with_expert.config.text_config.hidden_size
        )
        self.action_in_proj = nn.Linear(self.config.max_action_dim, self.vlm_with_expert.expert_hidden_size)
        self.action_out_proj = nn.Linear(self.vlm_with_expert.expert_hidden_size, self.config.max_action_dim)

        # Action time MLPs
        self.action_time_mlp_in = nn.Linear(
            self.vlm_with_expert.expert_hidden_size * 2, self.vlm_with_expert.expert_hidden_size
        )
        self.action_time_mlp_out = nn.Linear(
            self.vlm_with_expert.expert_hidden_size, self.vlm_with_expert.expert_hidden_size
        )

        self.set_requires_grad()

        # Special tokens
        self.fake_image_token = self.vlm_with_expert.processor.tokenizer.fake_image_token_id
        self.global_image_token = self.vlm_with_expert.processor.tokenizer.global_image_token_id
        self.global_image_start_token = torch.tensor(
            [self.fake_image_token, self.global_image_token], dtype=torch.long
        )
        self.image_end_token = torch.tensor([self.fake_image_token], dtype=torch.long)

        self.add_image_special_tokens = self.config.add_image_special_tokens
        self.prefix_length = self.config.prefix_length

    def set_requires_grad(self):
        """Set requires_grad for parameters based on config."""
        for params in self.state_proj.parameters():
            params.requires_grad = self.config.train_state_proj

    def sample_noise(self, shape, device):
        """Sample noise for flow matching."""
        noise = torch.normal(
            mean=0.0,
            std=1.0,
            size=shape,
            dtype=torch.float32,
            device=device,
        )
        return noise

    def sample_time(self, bsize, device):
        """Sample time for flow matching."""
        beta_dist = torch.distributions.Beta(concentration1=1.5, concentration0=1.0)
        time_beta = beta_dist.sample((bsize,)).to(device=device, dtype=torch.float32)
        time = time_beta * 0.999 + 0.001
        return time

    def sample_actions(self, images, img_masks, lang_tokens, lang_masks, state, noise=None) -> Tensor:
        """Sample actions using flow matching."""
        bsize = state.shape[0]
        device = state.device

        if noise is None:
            actions_shape = (bsize, self.config.chunk_size, self.config.max_action_dim)
            noise = self.sample_noise(actions_shape, device)

        # For inference, we use a simplified approach
        # In a full implementation, this would use the flow matching sampling process
        # Here we use a single denoising step for simplicity
        time = torch.tensor(0.5, dtype=torch.float32, device=device).expand(bsize)
        
        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        
        # Simple action sampling - in full implementation this would be iterative
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(noise, time)
        
        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)
        
        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1
        
        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )
        
        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        actions = self.action_out_proj(suffix_out)
        
        return actions

    def embed_prefix(self, images, img_masks, lang_tokens, lang_masks, state: torch.Tensor = None):
        """Embed prefix (images, language, state) for processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Process images
        for _img_idx, (img, img_mask) in enumerate(zip(images, img_masks)):
            if self.add_image_special_tokens:
                image_start_token = (
                    self.vlm_with_expert.embed_language_tokens(
                        self.global_image_start_token.to(device=self.vlm_with_expert.vlm.device)
                    )
                    .unsqueeze(0)
                    .expand(img.shape[0], -1, -1)
                )
                image_start_mask = torch.ones_like(
                    image_start_token[:, :, 0], dtype=torch.bool, device=image_start_token.device
                )
                att_masks += [0] * (image_start_mask.shape[-1])
                embs.append(image_start_token)
                pad_masks.append(image_start_mask)

            img_emb = self.vlm_with_expert.embed_image(img)
            
            # Normalize image embeddings
            img_emb_dim = img_emb.shape[-1]
            img_emb = img_emb * torch.tensor(img_emb_dim**0.5, dtype=img_emb.dtype, device=img_emb.device)

            bsize, num_img_embs = img_emb.shape[:2]
            img_mask = img_mask[:, None].expand(bsize, num_img_embs)

            embs.append(img_emb)
            pad_masks.append(img_mask)
            att_masks += [0] * (num_img_embs)

        # Process language
        lang_emb = self.vlm_with_expert.embed_language_tokens(lang_tokens)
        lang_emb_dim = lang_emb.shape[-1]
        lang_emb = lang_emb * math.sqrt(lang_emb_dim)

        embs.append(lang_emb)
        pad_masks.append(lang_masks)
        num_lang_embs = lang_emb.shape[1]
        att_masks += [0] * num_lang_embs

        # Process state
        state_emb = self.state_proj(state)
        state_emb = state_emb[:, None, :] if state_emb.ndim == 2 else state_emb
        embs.append(state_emb)
        bsize = state_emb.shape[0]
        device = state_emb.device

        states_seq_len = state_emb.shape[1]
        state_mask = torch.ones(bsize, states_seq_len, dtype=torch.bool, device=device)
        pad_masks.append(state_mask)
        att_masks += [1] * (states_seq_len)

        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=torch.bool, device=pad_masks.device)
        att_masks = att_masks[None, :]

        seq_len = pad_masks.shape[1]
        if seq_len < self.prefix_length and self.prefix_length > 0:
            embs = pad_tensor(embs, self.prefix_length, pad_value=0)
            pad_masks = pad_tensor(pad_masks, self.prefix_length, pad_value=0)
            att_masks = pad_tensor(att_masks, self.prefix_length, pad_value=0)

        att_masks = att_masks.expand(bsize, -1)

        return embs, pad_masks, att_masks

    def embed_suffix(self, noisy_actions, timestep):
        """Embed suffix (actions + time) for processing."""
        embs = []
        pad_masks = []
        att_masks = []

        # Fuse timestep + action information using an MLP
        action_emb = self.action_in_proj(noisy_actions)
        device = action_emb.device
        bsize = action_emb.shape[0]
        dtype = action_emb.dtype

        # Embed timestep using sine-cosine positional encoding
        time_emb = create_sinusoidal_pos_embedding(
            timestep,
            self.vlm_with_expert.expert_hidden_size,
            self.config.min_period,
            self.config.max_period,
            device=device,
        )
        time_emb = time_emb.type(dtype=dtype)

        time_emb = time_emb[:, None, :].expand_as(action_emb)
        action_time_emb = torch.cat([action_emb, time_emb], dim=2)

        action_time_emb = self.action_time_mlp_in(action_time_emb)
        action_time_emb = F.silu(action_time_emb)
        action_time_emb = self.action_time_mlp_out(action_time_emb)

        embs.append(action_time_emb)

        bsize, action_time_dim = action_time_emb.shape[:2]
        action_time_mask = torch.ones(bsize, action_time_dim, dtype=torch.bool, device=device)
        pad_masks.append(action_time_mask)

        att_masks += [1] * self.config.chunk_size
        embs = torch.cat(embs, dim=1)
        pad_masks = torch.cat(pad_masks, dim=1)
        att_masks = torch.tensor(att_masks, dtype=embs.dtype, device=embs.device)
        att_masks = att_masks[None, :].expand(bsize, len(att_masks))

        return embs, pad_masks, att_masks

    def forward(self, images, img_masks, lang_tokens, lang_masks, state, actions, noise=None, time=None) -> Tensor:
        """Forward pass for training."""
        if noise is None:
            noise = self.sample_noise(actions.shape, actions.device)

        if time is None:
            time = self.sample_time(actions.shape[0], actions.device)

        time_expanded = time[:, None, None]
        x_t = time_expanded * noise + (1 - time_expanded) * actions
        u_t = noise - actions

        prefix_embs, prefix_pad_masks, prefix_att_masks = self.embed_prefix(
            images, img_masks, lang_tokens, lang_masks, state=state
        )
        suffix_embs, suffix_pad_masks, suffix_att_masks = self.embed_suffix(x_t, time)

        pad_masks = torch.cat([prefix_pad_masks, suffix_pad_masks], dim=1)
        att_masks = torch.cat([prefix_att_masks, suffix_att_masks], dim=1)

        att_2d_masks = make_att_2d_masks(pad_masks, att_masks)
        position_ids = torch.cumsum(pad_masks, dim=1) - 1

        (_, suffix_out), _ = self.vlm_with_expert.forward(
            attention_mask=att_2d_masks,
            position_ids=position_ids,
            past_key_values=None,
            inputs_embeds=[prefix_embs, suffix_embs],
            use_cache=False,
            fill_kv_cache=False,
        )

        suffix_out = suffix_out[:, -self.config.chunk_size :]
        suffix_out = suffix_out.to(dtype=torch.float32)
        v_t = self.action_out_proj(suffix_out)
        losses = F.mse_loss(u_t, v_t, reduction="none")

        return losses
