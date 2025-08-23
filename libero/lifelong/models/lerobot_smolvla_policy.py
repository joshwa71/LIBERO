import os
import torch
import torch.nn as nn
import numpy as np
from pathlib import Path

from libero.lifelong.models.base_policy import BasePolicy
from libero.lifelong.models.smolvla.modeling_smolvla import SmolVLAPolicy

# quat_to_axis_angle function implementation to resolve the ModuleNotFoundError
def quat_to_axis_angle(quat):
    """
    Converts a quaternion to an axis-angle representation.
    
    Args:
        quat (torch.Tensor): A tensor of shape (..., 4) representing quaternions in (x, y, z, w) format.
        
    Returns:
        torch.Tensor: A tensor of shape (..., 3) representing the axis-angle vector.
    """
    # Ensure w is the last component for calculations
    xyz, w = quat[..., :3], quat[..., 3:]
    
    # Calculate the angle
    angle = 2 * torch.acos(torch.clamp(w, -1.0, 1.0))
    
    # Calculate the axis
    sin_half_angle = torch.sin(angle / 2)
    
    # Avoid division by zero for identity rotations (angle is zero)
    # A small epsilon is used for numerical stability
    axis = torch.where(
        torch.abs(sin_half_angle) > 1e-6,
        xyz / sin_half_angle,
        torch.zeros_like(xyz)
    )
    
    # The final axis-angle vector is axis * angle
    return axis * angle


class LerobotSmolVLAPolicy(BasePolicy):
    """
    A LIBERO-compatible wrapper for a pretrained Lerobot SmolVLA policy.
    
    This wrapper handles the conversion between LIBERO's data format and 
    Lerobot's expected input format, allowing evaluation of trained SmolVLA 
    policies within the LIBERO framework.
    """
    
    def __init__(self, pretrained_model_path, cfg, shape_meta):
        """
        Initialize the Lerobot SmolVLA policy wrapper.
        
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
        
        # Store task instruction - will be set by evaluation script
        self.current_task_instruction = None
        
        # Load the pretrained SmolVLA policy
        print(f"Loading SmolVLA policy from: {self.pretrained_model_path}")
        self._load_smolvla_policy()
        
        # Initialize state for LIBERO interface
        self.reset()
    
    def _load_smolvla_policy(self):
        """Load the pretrained SmolVLA policy from the specified path."""
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
            # Load the policy using SmolVLA's from_pretrained method
            self.smolvla_policy = SmolVLAPolicy.from_pretrained(str(self.pretrained_model_path))
            self.smolvla_policy.to(self.device)
            self.smolvla_policy.eval()
            
            print(f"Successfully loaded SmolVLA policy with:")
            print(f"  Input features: {list(self.smolvla_policy.config.input_features.keys())}")
            print(f"  Output features: {list(self.smolvla_policy.config.output_features.keys())}")
            print(f"  Device: {self.device}")
            
        except Exception as e:
            raise RuntimeError(f"Failed to load SmolVLA policy: {e}")
    
    def set_task_instruction(self, task_instruction):
        """
        Set the task instruction for SmolVLA.
        
        Args:
            task_instruction (str): The task instruction in natural language
        """
        self.current_task_instruction = task_instruction
        print(f"Task instruction set to: {task_instruction}")
    
    def _convert_libero_to_lerobot_obs(self, data):
        """
        Convert LIBERO observation format to Lerobot format.
        """
        obs = data["obs"]

        # Extract all necessary components
        agentview_rgb = obs["agentview_rgb"]
        eye_in_hand_rgb = obs["eye_in_hand_rgb"]
        gripper_states = obs["gripper_states"]
        joint_states = obs["joint_states"]
        ee_pos = obs["ee_pos"]
        ee_ori_quat = obs["ee_ori"]

        # Handle temporal dimension - take the last timestep if needed
        if agentview_rgb.dim() == 5:
            agentview_rgb, eye_in_hand_rgb, gripper_states, joint_states, ee_pos, ee_ori_quat = [
                x[:, -1] for x in [agentview_rgb, eye_in_hand_rgb, gripper_states, joint_states, ee_pos, ee_ori_quat]
            ]
        
        # Convert 4D quaternion to 3D axis-angle representation
        # The training data was created with a 3D axis-angle representation for orientation.
        ee_ori_axis_angle = quat_to_axis_angle(ee_ori_quat)

        # Concatenate to form the 15D observation state, now with correct components
        obs_state = torch.cat([ee_pos, ee_ori_axis_angle, gripper_states, joint_states], dim=-1)

        # Image normalization
        agentview_rgb, eye_in_hand_rgb = agentview_rgb.float(), eye_in_hand_rgb.float()
        if agentview_rgb.max() > 1.0: agentview_rgb /= 255.0
        if eye_in_hand_rgb.max() > 1.0: eye_in_hand_rgb /= 255.0
        agentview_rgb = agentview_rgb * 2.0 - 1.0
        eye_in_hand_rgb = eye_in_hand_rgb * 2.0 - 1.0

        # Task instruction
        task_instruction = self.current_task_instruction
        if task_instruction is None:
            if isinstance(data, dict) and 'task_str' in data:
                task_instruction = data['task_str']
            elif isinstance(data, dict) and 'task' in data and isinstance(data['task'], str):
                task_instruction = data['task']
            else:
                task_instruction = "Complete the task"
                print("WARNING: No task instruction found, using generic fallback")

        return {
            "observation.state": obs_state,
            "observation.images.top": agentview_rgb,
            "observation.images.wrist": eye_in_hand_rgb,
            "task": task_instruction
        }

    def get_action(self, data):
        """
        Get action from the SmolVLA policy in LIBERO format.
        
        Args:
            data (dict): LIBERO observation format
            
        Returns:
            np.ndarray: Action in LIBERO format (B, 7)
        """
        self.eval()
        
        with torch.no_grad():
            # Convert LIBERO format to Lerobot format
            lerobot_obs = self._convert_libero_to_lerobot_obs(data)

            # Get action from SmolVLA policy
            action_tensor = self.smolvla_policy.select_action(lerobot_obs)
            
            # Convert to numpy for LIBERO
            action_np = action_tensor.detach().cpu().numpy()
            
            # Ensure proper shape for LIBERO
            if action_np.ndim == 1:
                # If 1D tensor of shape (7,), reshape to (1, 7)
                action_np = action_np.reshape(1, -1)
            elif action_np.ndim == 2:
                # SmolVLA outputs actions with shape (batch_size, action_dim)
                # LIBERO expects (batch_size, 7) for Panda robot
                if action_np.shape[1] > 7:
                    # Only slice if there are more than 7 dimensions
                    action_np = action_np[:, :7]
            else:
                raise ValueError(f"Unexpected action shape: {action_np.shape}")
                
            return action_np
    
    def reset(self):
        """Reset the policy state (required by LIBERO interface)."""
        if hasattr(self, 'smolvla_policy'):
            self.smolvla_policy.reset()
    
    def forward(self, data):
        """
        Forward pass for training (not used in evaluation but required by interface).
        
        Args:
            data: Input data
            
        Returns:
            NotImplementedError: This wrapper is for evaluation only
        """
        raise NotImplementedError("LerobotSmolVLAPolicy is for evaluation only. Training is not supported.")
    
    def compute_loss(self, data, reduction="mean"):
        """
        Compute loss for training (not used in evaluation but required by interface).
        
        Args:
            data: Input data
            reduction: Loss reduction method
            
        Returns:
            NotImplementedError: This wrapper is for evaluation only
        """
        raise NotImplementedError("LerobotSmolVLAPolicy is for evaluation only. Training is not supported.")