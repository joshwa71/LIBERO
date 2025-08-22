# Integrating Lerobot ACT Policy Evaluation in LIBERO

## Overview

This document summarizes the successful integration of a pretrained Lerobot ACT (Action Chunking Transformer) policy into the LIBERO benchmark framework for evaluation. The project bridges two different robotics frameworks with incompatible dependencies and data formats.

## Project Context

- **Objective**: Evaluate a Lerobot ACT policy trained on LIBERO data within the LIBERO evaluation framework
- **Challenge**: Lerobot (torch 2.7.1) and LIBERO (torch 1.11.0) have incompatible dependencies
- **Training Setup**: ACT policy was trained on "turn on the stove and put the moka pot on it" task using converted LIBERO data

## Approach Taken

### Initial Approach (Abandoned)
- **Strategy**: Import Lerobot components directly into LIBERO
- **Issue**: Dependency conflicts between torch versions and other libraries
- **Problem**: `ModuleNotFoundError: No module named 'robomimic'` and version incompatibilities

### Final Approach (Successful)
- **Strategy**: Self-contained implementation by copying necessary Lerobot components into LIBERO
- **Benefits**: 
  - No external dependencies on Lerobot
  - Full compatibility with LIBERO's torch 1.11.0 environment
  - Maintainable and isolated implementation

## Technical Implementation

### 1. ACT Model Components
Created a complete ACT implementation within LIBERO:
- **Core Model**: Full ACT architecture with transformer encoder/decoder
- **Configuration**: JSON-based config loading with filtering for LIBERO compatibility
- **Normalization**: Data normalization/unnormalization for model input/output
- **Model Loading**: SafeTensors-based pretrained model loading

### 2. Data Format Conversion
Implemented bidirectional data format conversion:

#### LIBERO → ACT Format:
```
LIBERO obs: {
    "obs": {
        "agentview_rgb": (B, C, H, W),
        "eye_in_hand_rgb": (B, C, H, W), 
        "gripper_states": (B, 2),
        "joint_states": (B, 7)
    },
    "task_emb": (B, embed_dim)
}

ACT obs: {
    "observation.state": (B, 15),           # [ee_pos(3) + ee_ori(3) + gripper(2) + joints(7)]
    "observation.images.top": (B, 3, 128, 128),
    "observation.images.wrist": (B, 3, 128, 128)
}
```

#### ACT → LIBERO Format:
- Convert tensor actions to numpy arrays
- Handle batch dimensions correctly

### 3. Framework Integration
- **Policy Wrapper**: `LerobotACTPolicy` implementing LIBERO's `BasePolicy` interface
- **Algorithm Wrapper**: `LerobotACTAlgorithm` for LIBERO's evaluation framework
- **Evaluation Script**: Configurable evaluation with rendering support

## Challenges and Solutions

### Challenge 1: Dependency Conflicts
- **Problem**: Lerobot (torch 2.7.1) vs LIBERO (torch 1.11.0)
- **Solution**: Self-contained implementation, copied ~1200 lines of ACT code

### Challenge 2: Python Version Compatibility
- **Problem**: Lerobot uses Python 3.10+ type annotations (`dict[str, Type]`)
- **Solution**: Converted to Python 3.8 compatible typing (`Dict[str, Type]`)

### Challenge 3: Configuration Loading
- **Problem**: Lerobot config contains keys unsupported by simplified ACT config
- **Solution**: Filtered config dictionary to only include supported keys

### Challenge 4: Module Initialization Order
- **Problem**: `cannot assign module before Module.__init__() call`
- **Solution**: Proper initialization order in wrapper classes

### Challenge 5: Data Format Mismatch
- **Problem**: LIBERO expects 15D state vector with specific ordering
- **Solution**: Used zeros for ee_pos/ee_ori to match training data conversion

### Challenge 6: Rendering Support
- **Problem**: LIBERO uses `OffScreenRenderEnv` by default (headless)
- **Solution**: Created `OnScreenRenderEnv` and conditional OpenGL platform settings

### Challenge 7: Torch Deprecation Warnings
- **Problem**: `__floordiv__` deprecated in torch operations
- **Solution**: Used `torch.div(..., rounding_mode='trunc')`

## Files Created

### Core ACT Implementation (`LIBERO/libero/lifelong/models/act/`)
- `__init__.py` - Module exports
- `modeling_act.py` - Complete ACT model (adapted from Lerobot)
- `configuration_act.py` - ACT configuration with JSON loading
- `normalize.py` - Data normalization utilities
- `types.py` - Type definitions (FeatureType, NormalizationMode, PolicyFeature)
- `constants.py` - Constants (ACTION, OBS_IMAGES, etc.)

### Integration Components
- `lerobot_act_policy.py` - LIBERO-compatible ACT policy wrapper
- `lerobot_act_algo.py` - LIBERO algorithm wrapper for evaluation
- `metric_render.py` - Custom evaluation function with rendering support

### Environment Extensions
- `OnScreenRenderEnv` class in `env_wrapper.py` - Rendering-enabled environment

### Evaluation Infrastructure
- `evaluate_lerobot_act.py` - Configurable evaluation script

## Files Modified

### Import Updates
- `LIBERO/libero/lifelong/models/__init__.py` - Added ACT policy imports
- `LIBERO/libero/lifelong/algos/__init__.py` - Added ACT algorithm import
- `LIBERO/libero/libero/envs/__init__.py` - Added OnScreenRenderEnv import

### Environment Extensions
- `LIBERO/libero/libero/envs/env_wrapper.py` - Added OnScreenRenderEnv class

## Key Dependencies Added
- `safetensors` - For loading pretrained model weights
- `einops` - For tensor reshaping operations (ACT requirement)

## Usage

### Basic Evaluation
```bash
python evaluate_lerobot_act.py \
  --model_path /path/to/pretrained_model \
  --task_id 2 \
  --benchmark libero_10 \
  --n_eval 20 \
  --device cuda
```

### With Rendering
```bash
python evaluate_lerobot_act.py \
  --model_path /path/to/pretrained_model \
  --task_id 2 \
  --benchmark libero_10 \
  --n_eval 20 \
  --device cuda \
  --render
```

### Task ID Reference
For the trained task "turn on the stove and put the moka pot on it":
- **Benchmark**: `libero_10`
- **Task ID**: `2`

## Technical Achievements

1. ✅ **Self-contained ACT Implementation** - No external Lerobot dependencies
2. ✅ **Cross-framework Data Conversion** - Seamless LIBERO ↔ ACT format translation
3. ✅ **Pretrained Model Loading** - SafeTensors-based model loading with statistics
4. ✅ **Evaluation Integration** - Full LIBERO evaluation framework compatibility
5. ✅ **Rendering Support** - Optional on-screen visualization
6. ✅ **Python 3.8 Compatibility** - Works with LIBERO's older Python version
7. ✅ **Torch 1.11.0 Compatibility** - Adapted for older PyTorch version

## Architecture Overview

```
┌────────────────────────────────────────────────────────────────┐
│                    LIBERO Environment                          │
│                                                                │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │ LIBERO Task     │    │ LerobotACTAlgorithm              │   │
│  │ - Task ID: 2    │───▶│ ┌──────────────────────────────┐ │   │
│  │ - Benchmark:    │    │ │ LerobotACTPolicy             │ │   │
│  │   libero_10     │    │ │ ┌──────────────────────────┐ │ │   │
│  └─────────────────┘    │ │ │ ACTPolicy (self-contained│ │ │   │
│                         │ │ │ - modeling_act.py        │ │ │   │
│  ┌─────────────────┐    │ │ │ - configuration_act.py   │ │ │   │
│  │ LIBERO Data     │    │ │ │ - normalize.py           │ │ │   │
│  │ Format          │───▶│ │ └──────────────────────────┘ │ │   │
│  │ - obs dict      │    │ │ Data Format Conversion       │ │   │
│  │ - task_emb      │    │ └──────────────────────────────┘ │   │
│  └─────────────────┘    └──────────────────────────────────┘   │
│                                                                │
│  ┌─────────────────┐    ┌──────────────────────────────────┐   │
│  │ OnScreenRenderEnv│   │ evaluate_lerobot_act.py          │   │
│  │ - Visualization │◀───│ - Configurable evaluation        │   │
│  │ - has_renderer  │    │ - Rendering support              │   │
│  └─────────────────┘    └──────────────────────────────────┘   │
└────────────────────────────────────────────────────────────────┘
```

## Success Metrics

- **Integration**: ✅ Lerobot ACT policy successfully loads and runs in LIBERO
- **Performance**: Evaluation completes successfully (7.5 seconds for 2 episodes)
- **Compatibility**: No dependency conflicts with LIBERO environment
- **Extensibility**: Framework can be extended for other Lerobot policies

## Conclusion

Successfully created a robust bridge between Lerobot and LIBERO frameworks, enabling evaluation of state-of-the-art ACT policies within LIBERO's comprehensive benchmark suite. The self-contained approach ensures maintainability and eliminates dependency conflicts, providing a solid foundation for cross-framework robotics research.
