"""
MotionAGFormer wrapper for 2D → 3D pose lifting.

Repo: https://github.com/TaatiTeam/MotionAGFormer
Confirmed against: model/MotionAGFormer.py, data/reader/h36m.py, data/const.py,
                   train.py, configs/h36m/MotionAGFormer-base.yaml

Key facts:
- Model class: MotionAGFormer in model/MotionAGFormer.py (not common/).
- Input shape:  [B, T, 17, 3]  where 3 = (x_norm, y_norm, confidence)
- Output shape: [B, T, 17, 3]  3D joint positions in same normalized space
- Normalization (from h36m.py): x = px/W*2 - 1,  y = py/W*2 - H/W
    Both axes divided by IMAGE WIDTH, not height.
- Confidence channel: 1.0 for all joints when not available from detector.
- H36M joint order (from data/const.py):
    0 sacrum  1 left_hip   2 left_knee   3 left_foot
    4 right_hip  5 right_knee  6 right_foot
    7 center_torso  8 upper_torso  9 neck_base  10 center_head
    11 right_shoulder  12 right_elbow  13 right_hand
    14 left_shoulder   15 left_elbow   16 left_hand
- Checkpoint structure: {'model': state_dict, 'args': Namespace, ...}
    Loaded with strict=True; args embedded in checkpoint define architecture.
"""

import importlib.util
import os
import sys
import types
from typing import Optional

import numpy as np
import torch

VENDOR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vendor"
)
MOTIONAGFORMER_DIR = os.path.join(VENDOR_DIR, "MotionAGFormer")


def _load_motionagformer_class(vendor_dir: str):
    """
    Load the MotionAGFormer class by absolute file path, bypassing sys.path and
    sys.modules entirely.  Required because other custom nodes often register a
    competing 'model' package in sys.modules before we run.
    """
    model_dir   = os.path.join(vendor_dir, "model")
    modules_dir = os.path.join(model_dir,  "modules")

    def _load(key: str, filepath: str):
        """Load a .py file and register it under `key` in sys.modules."""
        if key in sys.modules:
            return sys.modules[key]
        spec = importlib.util.spec_from_file_location(key, filepath)
        mod  = importlib.util.module_from_spec(spec)
        sys.modules[key] = mod          # register BEFORE exec so circular imports work
        spec.loader.exec_module(mod)
        return mod

    # Build synthetic package stubs so sub-imports like
    # "from model.modules.attention import Attention" resolve correctly.
    for pkg_key, pkg_path in [
        ("model",         model_dir),
        ("model.modules", modules_dir),
    ]:
        if pkg_key not in sys.modules:
            pkg = types.ModuleType(pkg_key)
            pkg.__path__    = [pkg_path]
            pkg.__package__ = pkg_key
            sys.modules[pkg_key] = pkg

    # Load each file in model/modules/
    for name in ("attention", "graph", "mlp", "tcn",
                 "metaformer", "normalization", "ctrgc", "ctr_attention"):
        fp = os.path.join(modules_dir, name + ".py")
        if os.path.isfile(fp):
            _load(f"model.modules.{name}", fp)

    # Load model/MotionAGFormer.py itself
    maf_mod = _load("model.MotionAGFormer",
                    os.path.join(model_dir, "MotionAGFormer.py"))
    return maf_mod.MotionAGFormer


def _ensure_in_path():
    if not os.path.isdir(MOTIONAGFORMER_DIR):
        raise RuntimeError(
            f"MotionAGFormer repo not found at {MOTIONAGFORMER_DIR}.\n"
            "Run: git clone https://github.com/TaatiTeam/MotionAGFormer vendor/MotionAGFormer"
        )
    if MOTIONAGFORMER_DIR not in sys.path:
        sys.path.insert(0, MOTIONAGFORMER_DIR)


class MotionAGFormerModel:
    """
    Wraps MotionAGFormer for single-frame or temporal 2D → 3D pose lifting.

    Input:  normalized 2D poses  [T, 17, 3]  (x_norm, y_norm, conf)
    Output: 3D joint positions   [T, 17, 3]  in the same normalized space
    """

    NUM_JOINTS = 17

    def __init__(self):
        self.model = None
        self.device: Optional[torch.device] = None
        self.n_frames: int = 1

    def load(self, checkpoint_path: str, device: str = "cuda", n_frames: int = 1):
        """
        Args:
            checkpoint_path: path to MotionAGFormer checkpoint (.bin / .pth)
            device:          'cuda' | 'cpu'
            n_frames:        temporal window the checkpoint was trained with.
                             Common values: 1, 27, 81, 243.
                             Must match what the checkpoint expects (positional
                             embeddings are sized to n_frames).
        """
        _ensure_in_path()

        MotionAGFormer = _load_motionagformer_class(MOTIONAGFORMER_DIR)

        self.device = torch.device(device)
        self.n_frames = n_frames

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        ckpt_args = checkpoint.get("args", None)

        # Resolve state dict and strip DataParallel "module." prefix early
        # so we can infer architecture from weights when ckpt_args is absent.
        state_dict = checkpoint.get("model", checkpoint)
        has_module_prefix = bool(state_dict and all(k.startswith("module.") for k in state_dict.keys()))
        if has_module_prefix:
            state_dict = {k[len("module."):]: v for k, v in state_dict.items()}

        # Diagnostic: log first 5 keys so we can see the actual prefix in ComfyUI logs
        _first_keys = list(state_dict.keys())[:5] if state_dict else []
        print(f"[MotionAGFormer] ckpt_args={ckpt_args is not None}, "
              f"has_module_prefix={has_module_prefix}, "
              f"first_keys={_first_keys}")

        def _infer_n_layers(sd):
            layer_keys = [k for k in sd if k.startswith("layers.")]
            if not layer_keys:
                return 16
            return max(int(k.split(".")[1]) for k in layer_keys) + 1

        def _infer_n_frames(sd):
            bn_keys = [k for k in sd if "graph_temporal.mixer.batch_norm.weight" in k]
            if not bn_keys:
                return n_frames
            return sd[bn_keys[0]].shape[0]

        if ckpt_args is not None:
            # Build from the args embedded in the checkpoint
            actual_n_frames = getattr(ckpt_args, "n_frames", _infer_n_frames(state_dict))
            self.n_frames = actual_n_frames
            self.model = MotionAGFormer(
                n_layers=getattr(ckpt_args, "n_layers", _infer_n_layers(state_dict)),
                dim_in=getattr(ckpt_args, "dim_in", 3),
                dim_feat=getattr(ckpt_args, "dim_feat", 128),
                dim_rep=getattr(ckpt_args, "dim_rep", 512),
                dim_out=3,
                mlp_ratio=getattr(ckpt_args, "mlp_ratio", 4),
                act_layer=torch.nn.GELU,
                attn_drop=0.0,
                drop=0.0,
                drop_path=0.0,
                use_layer_scale=getattr(ckpt_args, "use_layer_scale", True),
                layer_scale_init_value=getattr(ckpt_args, "layer_scale_init_value", 1e-5),
                use_adaptive_fusion=getattr(ckpt_args, "use_adaptive_fusion", True),
                num_heads=getattr(ckpt_args, "num_heads", 8),
                qkv_bias=getattr(ckpt_args, "qkv_bias", False),
                qkv_scale=None,
                hierarchical=getattr(ckpt_args, "hierarchical", False),
                num_joints=self.NUM_JOINTS,
                use_temporal_similarity=getattr(ckpt_args, "use_temporal_similarity", True),
                temporal_connection_len=getattr(ckpt_args, "temporal_connection_len", 1),
                use_tcn=getattr(ckpt_args, "use_tcn", False),
                graph_only=getattr(ckpt_args, "graph_only", False),
                neighbour_num=getattr(ckpt_args, "neighbour_num", 2),
                n_frames=actual_n_frames,
            )
        else:
            # No args in checkpoint — infer architecture from state dict weights
            actual_n_frames = _infer_n_frames(state_dict)
            actual_n_layers = _infer_n_layers(state_dict)
            print(f"[MotionAGFormer] inferred n_layers={actual_n_layers}, n_frames={actual_n_frames}")
            self.n_frames = actual_n_frames
            self.model = MotionAGFormer(
                n_layers=actual_n_layers,
                dim_in=3,
                dim_feat=128,
                dim_rep=512,
                dim_out=3,
                num_heads=8,
                mlp_ratio=4,
                hierarchical=False,
                use_temporal_similarity=True,
                neighbour_num=2,
                use_tcn=False,
                graph_only=False,
                num_joints=self.NUM_JOINTS,
                n_frames=actual_n_frames,
            )

        self.model.load_state_dict(state_dict, strict=True)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def lift(self, poses_2d: np.ndarray) -> np.ndarray:
        """
        Lift a sequence of normalized 2D poses to 3D.

        Args:
            poses_2d: (T, 17, 3) float32
                      [x_norm, y_norm, confidence] per joint, per frame
                      Normalization: x = px/W*2 - 1,  y = py/W*2 - H/W

        Returns:
            poses_3d: (T, 17, 3) float32  in the same normalized space
        """
        if self.model is None:
            raise RuntimeError("Call load() first.")

        T = poses_2d.shape[0]

        # Pad / trim to match checkpoint's n_frames
        if T < self.n_frames:
            pad = np.tile(poses_2d[[-1]], (self.n_frames - T, 1, 1))
            inp = np.concatenate([poses_2d, pad], axis=0)
        else:
            inp = poses_2d[-self.n_frames :]

        x = torch.from_numpy(inp).float().unsqueeze(0).to(self.device)  # (1, T, 17, 3)
        out = self.model(x)  # (1, T, 17, 3)
        poses_3d = out.squeeze(0).cpu().numpy()  # (T, 17, 3)
        return poses_3d[:T]
