import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.act.modeling_act import ACTPolicy


class LerobotACTPolicy(BasePolicy):
    """
    A LIBERO-compatible wrapper for a pretrained Lerobot ACT policy.
    
    This wrapper handles the conversion between LIBERO's data format and 
    Lerobot's expected input format, allowing evaluation of trained ACT 
    policies within the LIBERO framework.
    """
    
    def __init__(self, pretrained_model_path, cfg, shape_meta):
        """
        Initialize the Lerobot ACT policy wrapper.
        
        Args:
            pretrained_model_path (str): Path to the pretrained model directory
            cfg: LIBERO configuration object
            shape_meta: LIBERO shape metadata (not used but required by interface)
        """
        # Call nn.Module.__init__ first
        nn.Module.__init__(self)
        
        self.cfg = cfg
        self.device = cfg.device
        self.shape_meta = shape_meta
        self.pretrained_model_path = Path(pretrained_model_path)
        
        # Load the pretrained ACT policy
        print(f"Loading ACT policy from: {self.pretrained_model_path}")
        self._load_act_policy()
        
        # Initialize state for LIBERO interface
        self.reset()
    
    def _load_act_policy(self):
        """Load the pretrained ACT policy from the specified path."""
        if not self.pretrained_model_path.exists():
            raise FileNotFoundError(f"Pretrained model path does not exist: {self.pretrained_model_path}")
        
        # Check for required files
        model_file = self.pretrained_model_path / "model.safetensors"
        config_file = self.pretrained_model_path / "config.json"
        
        if not model_file.exists():
            raise FileNotFoundError(f"Model file not found: {model_file}")
        if not config_file.exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")
            
        try:
            # Load the policy using Lerobot's from_pretrained method
            self.act_policy = ACTPolicy.from_pretrained(str(self.pretrained_model_path))
            self.act_policy.to(self.device)
            self.act_policy.eval()
            
            print(f"Successfully loaded ACT policy with:")
            print(f"  Input features: {list(self.act_policy.config.input_features.keys())}")
            print(f"  Output features: {list(self.act_policy.config.output_features.keys())}")
            print(f"  Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load ACT policy: {e}")
    
    def _convert_libero_to_lerobot_obs(self, data):
        """
        Convert LIBERO observation format to Lerobot format.
        
        Args:
            data (dict): LIBERO data format with keys:
                - obs: dict with agentview_rgb, eye_in_hand_rgb, gripper_states, joint_states
                - task_emb: task embedding (ignored for ACT)
                
        Returns:
            dict: Lerobot observation format
        """
        obs = data["obs"]
        
        # Extract components - handle both (B, T, ...) and (B, ...) cases
        agentview_rgb = obs["agentview_rgb"]
        eye_in_hand_rgb = obs["eye_in_hand_rgb"] 
        gripper_states = obs["gripper_states"]
        joint_states = obs["joint_states"]
        
        # Handle temporal dimension - take the last timestep if needed
        if agentview_rgb.dim() == 5:  # (B, T, C, H, W)
            agentview_rgb = agentview_rgb[:, -1]  # Take last timestep
        if eye_in_hand_rgb.dim() == 5:  # (B, T, C, H, W)
            eye_in_hand_rgb = eye_in_hand_rgb[:, -1]
        if gripper_states.dim() == 3:  # (B, T, D)
            gripper_states = gripper_states[:, -1]
        if joint_states.dim() == 3:  # (B, T, D)
            joint_states = joint_states[:, -1]
        
        # For observation.state, we need to reconstruct the 15D vector as:
        # [ee_pos(3), ee_ori(3), gripper_states(2), joint_states(7)]
        # The trained model expects this specific concatenation order
        
        batch_size = joint_states.shape[0]
        device = joint_states.device
        
        # Extract end-effector position and orientation from robot state
        # The trained model was converted with zeros for ee_pos and ee_ori (as placeholders)
        # We need to match this exactly to maintain compatibility
        
        # Use zeros for ee_pos and ee_ori to match the conversion script behavior
        ee_pos = torch.zeros(batch_size, 3, device=device, dtype=joint_states.dtype)
        ee_ori = torch.zeros(batch_size, 3, device=device, dtype=joint_states.dtype)
        
        # Concatenate to form the 15D observation state exactly as expected by the trained model
        obs_state = torch.cat([ee_pos, ee_ori, gripper_states, joint_states], dim=-1)
        
        # Convert images from LIBERO format (B, C, H, W) to Lerobot format (B, C, H, W)
        # Images should already be in the right format, but ensure they're float32 and in [0,1]
        agentview_rgb = agentview_rgb.float()
        eye_in_hand_rgb = eye_in_hand_rgb.float() 
        
        # If images are in [0, 255], normalize to [0, 1]
        if agentview_rgb.max() > 1.0:
            agentview_rgb = agentview_rgb / 255.0
        if eye_in_hand_rgb.max() > 1.0:
            eye_in_hand_rgb = eye_in_hand_rgb / 255.0
        
        return {
            "observation.state": obs_state,
            "observation.images.top": agentview_rgb,
            "observation.images.wrist": eye_in_hand_rgb
        }
    
    def get_action(self, data):
        """
        Get action from the ACT policy in LIBERO format.
        
        Args:
            data (dict): LIBERO observation format
            
        Returns:
            np.ndarray: Action in LIBERO format (B, 7)
        """
        self.eval()
        
        with torch.no_grad():
            # Convert LIBERO format to Lerobot format
            lerobot_obs = self._convert_libero_to_lerobot_obs(data)
            
            # Get action from ACT policy
            action_tensor = self.act_policy.select_action(lerobot_obs)
            
            # Convert to numpy for LIBERO
            action_np = action_tensor.detach().cpu().numpy()
            
            # Ensure proper shape for LIBERO (B, 7)
            if action_np.ndim == 1:
                action_np = action_np.reshape(1, -1)
                
            return action_np
    
    def reset(self):
        """Reset the policy state (required by LIBERO interface)."""
        if hasattr(self, 'act_policy'):
            self.act_policy.reset()
    
    def forward(self, data):
        """
        Forward pass for training (not used in evaluation but required by interface).
        
        Args:
            data: Input data
            
        Returns:
            NotImplementedError: This wrapper is for evaluation only
        """
        raise NotImplementedError("LerobotACTPolicy is for evaluation only. Training is not supported.")
    
    def compute_loss(self, data, reduction="mean"):
        """
        Compute loss for training (not used in evaluation but required by interface).
        
        Args:
            data: Input data
            reduction: Loss reduction method
            
        Returns:
            NotImplementedError: This wrapper is for evaluation only
        """
        raise NotImplementedError("LerobotACTPolicy is for evaluation only. Training is not supported.")
