#!/usr/bin/env python3
"""
Evaluation script for Lerobot SmolVLA policies in LIBERO environments.

This script allows evaluation of pretrained Lerobot SmolVLA policies on LIBERO tasks
with configurable options for rendering, task selection, and evaluation parameters.

Usage:
    python evaluate_lerobot_smolvla.py --model_path /path/to/model --task_id 0 --n_eval 10
"""

import argparse
import os
import sys
import time
import torch
import numpy as np
from pathlib import Path

# Add LIBERO to path
libero_path = Path(__file__).parent
sys.path.insert(0, str(libero_path))

from libero.libero import benchmark, get_libero_path
from libero.libero.benchmark import get_benchmark
from libero.lifelong.datasets import get_dataset
from libero.lifelong.utils import get_task_embs, safe_device
from libero.lifelong.algos.lerobot_smolvla_algo import LerobotSmolVLAAlgorithm
from libero.lifelong.metric import evaluate_one_task_success
from libero.lifelong.metric_render import evaluate_one_task_success_with_rendering

# Import hydra for config management
try:
    from hydra import compose, initialize
    from hydra.core.global_hydra import GlobalHydra
    from omegaconf import OmegaConf
    import yaml
    from easydict import EasyDict
    HYDRA_AVAILABLE = True
except ImportError:
    print("Warning: Hydra not available, using minimal configuration")
    from easydict import EasyDict
    HYDRA_AVAILABLE = False

# Set environment variables
os.environ["TOKENIZERS_PARALLELISM"] = "false"
# Don't set PYOPENGL_PLATFORM to 'egl' when rendering is enabled
# This will be set conditionally based on render flag


def setup_libero_config(
    benchmark_name="libero_object",
    task_order_index=0,
    device="cuda",
    n_eval=20,
    max_steps=300,
    img_h=128,
    img_w=128,
    use_mp=True,
    num_procs=1,
    render=False,
    seed=10000,
    task_instruction=""
):
    """
    Setup LIBERO configuration for evaluation.
    
    Args:
        benchmark_name: LIBERO benchmark name
        task_order_index: Task order index (0-21)
        device: Device to use ('cuda' or 'cpu')
        n_eval: Number of evaluation episodes
        max_steps: Maximum steps per episode
        img_h: Image height
        img_w: Image width
        use_mp: Use multiprocessing
        num_procs: Number of processes
        render: Enable rendering
        seed: Random seed
        task_instruction: Task instruction for SmolVLA
        
    Returns:
        EasyDict: LIBERO configuration
    """
    # Initialize hydra config system or create minimal config
    if HYDRA_AVAILABLE:
        try:
            # Clear any existing hydra instance
            GlobalHydra.instance().clear()
            
            initialize(config_path="libero/configs", version_base=None)
            hydra_cfg = compose(config_name="config")
            yaml_config = OmegaConf.to_yaml(hydra_cfg)
            cfg = EasyDict(yaml.safe_load(yaml_config))
        except Exception as e:
            print(f"Warning: Could not load hydra config, using minimal config: {e}")
            cfg = EasyDict()
    else:
        print("Creating minimal config (hydra not available)")
        cfg = EasyDict()
    
    # Override with our settings
    cfg.benchmark_name = benchmark_name
    cfg.device = device
    cfg.seed = seed
    cfg.task_instruction = task_instruction  # Add task instruction for SmolVLA
    
    # Data config
    if not hasattr(cfg, 'data'):
        cfg.data = EasyDict()
    cfg.data.task_order_index = task_order_index
    cfg.data.img_h = img_h
    cfg.data.img_w = img_w
    cfg.data.seq_len = 10  # Default for LIBERO
    cfg.data.use_joint = True
    cfg.data.use_gripper = True
    cfg.data.use_ee = False
    
    # Observation configuration
    cfg.data.obs = EasyDict()
    cfg.data.obs.modality = EasyDict()
    cfg.data.obs.modality.rgb = ["agentview_rgb", "eye_in_hand_rgb"]
    cfg.data.obs.modality.depth = []
    cfg.data.obs.modality.low_dim = ["gripper_states", "joint_states"]
    
    # Observation key mapping (from LIBERO environment to LIBERO data format)
    cfg.data.obs_key_mapping = EasyDict()
    cfg.data.obs_key_mapping.agentview_rgb = "agentview_image"
    cfg.data.obs_key_mapping.eye_in_hand_rgb = "robot0_eye_in_hand_image"
    cfg.data.obs_key_mapping.gripper_states = "robot0_gripper_qpos"
    cfg.data.obs_key_mapping.joint_states = "robot0_joint_pos"
    
    # Evaluation config
    if not hasattr(cfg, 'eval'):
        cfg.eval = EasyDict()
    cfg.eval.n_eval = n_eval
    cfg.eval.max_steps = max_steps
    cfg.eval.use_mp = use_mp
    cfg.eval.num_procs = num_procs if use_mp else 1
    cfg.eval.render = render
    cfg.eval.save_sim_states = False
    
    # Set LIBERO paths
    cfg.folder = get_libero_path("datasets")
    cfg.bddl_folder = get_libero_path("bddl_files")
    cfg.init_states_folder = get_libero_path("init_states")
    
    # Lifelong learning config (minimal for evaluation)
    if not hasattr(cfg, 'lifelong'):
        cfg.lifelong = EasyDict()
    cfg.lifelong.algo = "LerobotSmolVLAAlgorithm"
    
    return cfg


def get_task_info(cfg, task_id):
    """
    Get task information and embeddings.
    
    Args:
        cfg: LIBERO configuration
        task_id: Task ID to evaluate
        
    Returns:
        tuple: (benchmark, task, task_emb, shape_meta)
    """
    # Get benchmark
    benchmark_instance = get_benchmark(cfg.benchmark_name)(cfg.data.task_order_index)
    n_tasks = benchmark_instance.n_tasks
    
    if task_id >= n_tasks:
        raise ValueError(f"Task ID {task_id} is out of range. Benchmark {cfg.benchmark_name} has {n_tasks} tasks.")
    
    # Get task and embedding
    task = benchmark_instance.get_task(task_id)
    
    # Update task instruction in config for SmolVLA
    cfg.task_instruction = task.language
    
    # Load a sample dataset to get shape metadata
    task_dataset_path = os.path.join(cfg.folder, benchmark_instance.get_task_demonstration(task_id))
    task_dataset, shape_meta = get_dataset(
        dataset_path=task_dataset_path,
        obs_modality=cfg.data.obs.modality,
        initialize_obs_utils=True,
        seq_len=cfg.data.seq_len,
    )
    
    # Get task embeddings for ALL tasks in the benchmark (required by LIBERO framework)
    descriptions = [benchmark_instance.get_task(i).language for i in range(n_tasks)]
    task_embs = get_task_embs(cfg, descriptions)
    benchmark_instance.set_task_embs(task_embs)
    
    # Now we can safely get the specific task embedding
    task_emb = benchmark_instance.get_task_emb(task_id)
    
    return benchmark_instance, task, task_emb, shape_meta


def evaluate_smolvla_policy(
    model_path,
    task_id=0,
    benchmark_name="libero_object",
    n_eval=20,
    max_steps=300,
    device="cuda",
    seed=10000,
    render=False,
    verbose=True
):
    """
    Evaluate a pretrained Lerobot SmolVLA policy on a LIBERO task.
    
    Args:
        model_path: Path to the pretrained SmolVLA model directory
        task_id: LIBERO task ID to evaluate on
        benchmark_name: LIBERO benchmark name
        n_eval: Number of evaluation episodes
        max_steps: Maximum steps per episode  
        device: Device to use
        seed: Random seed
        render: Enable rendering
        verbose: Print detailed information
        
    Returns:
        float: Success rate
    """
    # Set OpenGL platform based on rendering preference
    if render:
        # For on-screen rendering, don't use egl
        if 'PYOPENGL_PLATFORM' in os.environ:
            del os.environ['PYOPENGL_PLATFORM']
        print("Enabling on-screen rendering (removed PYOPENGL_PLATFORM=egl)")
    else:
        # For headless rendering, use egl
        os.environ['PYOPENGL_PLATFORM'] = 'egl'
    
    if verbose:
        print("=" * 60)
        print("LEROBOT SMOLVLA POLICY EVALUATION IN LIBERO")
        print("=" * 60)
        print(f"Model path: {model_path}")
        print(f"Task ID: {task_id}")
        print(f"Benchmark: {benchmark_name}")
        print(f"Episodes: {n_eval}")
        print(f"Max steps: {max_steps}")
        print(f"Device: {device}")
        print(f"Seed: {seed}")
        print(f"Render: {render}")
        print()
    
    # Setup configuration
    cfg = setup_libero_config(
        benchmark_name=benchmark_name,
        device=device,
        n_eval=n_eval,
        max_steps=max_steps,
        render=render,
        seed=seed
    )
    
    # Set random seeds
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Get task information
    if verbose:
        print("Setting up task...")
    benchmark_instance, task, task_emb, shape_meta = get_task_info(cfg, task_id)
    
    if verbose:
        print(f"Task: {task.language}")
        print(f"Problem folder: {task.problem_folder}")
        print(f"BDDL file: {task.bddl_file}")
        print()
    
    # Create algorithm with SmolVLA policy
    if verbose:
        print("Loading SmolVLA policy...")
    cfg.shape_meta = shape_meta  # Add shape_meta to config
    algo = LerobotSmolVLAAlgorithm(model_path, cfg, shape_meta)
    algo = safe_device(algo, device)
    
    if verbose:
        print("Starting evaluation...")
        print()
    
    # Run evaluation
    start_time = time.time()
    if render:
        success_rate = evaluate_one_task_success_with_rendering(
            cfg=cfg,
            algo=algo,
            task=task,
            task_emb=task_emb,
            task_id=task_id,
            sim_states=None,
            task_str=task.language,  # Pass task language for SmolVLA
            enable_rendering=True
        )
    else:
        success_rate = evaluate_one_task_success(
            cfg=cfg,
            algo=algo,
            task=task,
            task_emb=task_emb,
            task_id=task_id,
            sim_states=None,
            task_str=task.language  # Pass task language for SmolVLA
        )
    end_time = time.time()
    
    if verbose:
        print("=" * 60)
        print("EVALUATION RESULTS")
        print("=" * 60)
        print(f"Success Rate: {success_rate:.2%} ({int(success_rate * n_eval)}/{n_eval})")
        print(f"Evaluation Time: {end_time - start_time:.1f} seconds")
        print(f"Time per Episode: {(end_time - start_time) / n_eval:.1f} seconds")
        print("=" * 60)
    
    return success_rate


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Evaluate Lerobot SmolVLA policy in LIBERO environment"
    )
    
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the pretrained SmolVLA model directory"
    )
    
    parser.add_argument(
        "--task_id",
        type=int,
        default=0,
        help="LIBERO task ID to evaluate on (default: 0)"
    )
    
    parser.add_argument(
        "--benchmark",
        type=str,
        default="libero_object",
        choices=["libero_spatial", "libero_object", "libero_goal", "libero_10"],
        help="LIBERO benchmark name (default: libero_object)"
    )
    
    parser.add_argument(
        "--n_eval",
        type=int,
        default=20,
        help="Number of evaluation episodes (default: 20)"
    )
    
    parser.add_argument(
        "--max_steps", 
        type=int,
        default=300,
        help="Maximum steps per episode (default: 300)"
    )
    
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        choices=["cuda", "cpu"],
        help="Device to use (default: cuda)"
    )
    
    parser.add_argument(
        "--seed",
        type=int,
        default=10000,
        help="Random seed (default: 10000)"
    )
    
    parser.add_argument(
        "--render",
        action="store_true",
        help="Enable rendering during evaluation"
    )
    
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce output verbosity"
    )
    
    args = parser.parse_args()
    
    # Validate model path
    model_path = Path(args.model_path)
    if not model_path.exists():
        print(f"Error: Model path does not exist: {model_path}")
        sys.exit(1)
    
    if not (model_path / "model.safetensors").exists():
        print(f"Error: No model.safetensors found in {model_path}")
        sys.exit(1)
    
    if not (model_path / "config.json").exists():
        print(f"Error: No config.json found in {model_path}")
        sys.exit(1)
    
    # Run evaluation
    try:
        success_rate = evaluate_smolvla_policy(
            model_path=str(model_path),
            task_id=args.task_id,
            benchmark_name=args.benchmark,
            n_eval=args.n_eval,
            max_steps=args.max_steps,
            device=args.device,
            seed=args.seed,
            render=args.render,
            verbose=not args.quiet
        )
        
        if args.quiet:
            print(f"{success_rate:.4f}")
        
    except Exception as e:
        print(f"Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
