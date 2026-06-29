# EditPose

EditPose is a ComfyUI custom node pack for single-image motion capture:

- `ClickPose` detects and interactively corrects 2D keypoints.
- `MotionAGFormer` lifts the corrected 2D pose into 3D joints.
- `3D Pose Editor` lets you refine the lifted pose in an embedded viewer.

This repository has been refactored to follow the `ameliacode/comfyui-template`
package layout: packaged nodes under `nodes/`, install hooks via `install.py`
and `prestartup_script.py`, and frontend assets under `js/`.

## Repository Layout

```text
comfyui-mocap/
├── .github/
├── checkpoints/
├── js/
├── modules/
├── nodes/
├── vendor/
├── workflows/
├── __init__.py
├── install.py
├── prestartup_script.py
├── pyproject.toml
├── requirements.txt
└── comfy-env-root.toml
```

## Installation

### Preferred: comfy-env

```bash
python -m pip install comfy-env
comfy-env install --config comfy-env-root.toml
```

### Fallback: plain pip

```bash
python -m pip install -r requirements.txt
```

## Usage

1. Add a source image to your workflow.
2. Run `ClickPose` and use the editor button to correct 2D joints.
3. Feed the result into `MotionAGFormer` to generate a 3D pose.
4. Refine the lifted result in `3D Pose Editor`.

## Notes

- Model checkpoints are expected under `checkpoints/`.
- Third-party upstream code is vendored under `vendor/`.
- The package keeps legacy node exports for compatibility while adopting the template structure.
