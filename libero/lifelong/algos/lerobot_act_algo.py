import torch
import torch.nn as nn
from libero.lifelong.models.lerobot_act_policy import LerobotACTPolicy


class LerobotACTAlgorithm(nn.Module):
    """
    LIBERO algorithm wrapper for pretrained Lerobot ACT policies.
    
    This wrapper allows evaluation of pretrained Lerobot ACT policies
    within LIBERO's evaluation framework without requiring training.
    """
    
    def __init__(self, pretrained_model_path, cfg, shape_meta):
        """
        Initialize the Lerobot ACT algorithm wrapper.
        
        Args:
            pretrained_model_path (str): Path to the pretrained ACT model
            cfg: LIBERO configuration
            shape_meta: LIBERO shape metadata
        """
        # Call nn.Module.__init__ directly
        super().__init__()
        
        self.cfg = cfg
        self.n_tasks = 1  # Single task for evaluation
        self.current_task = -1
        self.pretrained_model_path = pretrained_model_path
        
        # Create the wrapped ACT policy
        self.policy = LerobotACTPolicy(pretrained_model_path, cfg, shape_meta)
        
        print(f"Initialized LerobotACTAlgorithm with policy from: {pretrained_model_path}")
    
    def start_task(self, task):
        """Start task (no-op for evaluation-only algorithm)."""
        self.current_task = task
        print(f"Started task {task} (evaluation mode)")
    
    def end_task(self, dataset, task_id, benchmark, env=None):
        """End task (no-op for evaluation-only algorithm)."""
        print(f"Ended task {task_id}")
    
    def observe(self, data):
        """
        Training step (not supported for evaluation-only algorithm).
        
        Args:
            data: Training data
            
        Returns:
            NotImplementedError: This algorithm is for evaluation only
        """
        raise NotImplementedError("LerobotACTAlgorithm is for evaluation only. Training is not supported.")
    
    def eval_observe(self, data):
        """
        Evaluation step (not used but may be called by LIBERO).
        
        Args:
            data: Evaluation data
            
        Returns:
            float: Dummy loss value
        """
        return 0.0
    
    def reset(self):
        """Reset the algorithm/policy state."""
        self.policy.reset()
    
    def learn_one_task(self, dataset, task_id, benchmark, result_summary):
        """
        Learn one task (no-op for evaluation-only algorithm).
        
        Returns dummy values to satisfy LIBERO's interface.
        """
        print(f"Skipping training for task {task_id} (evaluation-only mode)")
        return 1.0, 0.0  # Dummy success and loss values
