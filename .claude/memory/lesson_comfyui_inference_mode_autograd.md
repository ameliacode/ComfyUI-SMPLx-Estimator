---
name: lesson-comfyui-inference-mode-autograd
description: ComfyUI runs nodes under torch.inference_mode(); nodes needing autograd must opt out
metadata:
  type: project
---

ComfyUI executes every node's FUNCTION inside `torch.inference_mode()` (verified at
`/home/wswg3/github/ComfyUI/execution.py:720`). This is STRONGER than `torch.no_grad()`:
tensors created inside become "inference tensors" that cannot participate in autograd,
and a model built there has inference-tensor buffers that raise
"Inference tensors cannot be saved for backward" during `loss.backward()`.

**Symptom:** `RuntimeError: element 0 of tensors does not require grad and does not have
a grad_fn` (or "Inference tensors cannot be saved for backward") inside a custom node
that runs an optimizer — even though the same code works in a standalone pytest.

**Why:** the comfyui-mocap SMPLXFit/SMPLXEditor nodes run gradient-descent fitting
(SMPLify-X / IK) through `smplx.forward()`, which needs autograd.

**How to apply:** any code path that needs gradients inside a ComfyUI node must run
under `with torch.inference_mode(False), torch.enable_grad():`, AND the models it
differentiates through must be CREATED outside inference mode too (wrap
`smplx.create(...)` and VPoser `load_model(...)` in `with torch.inference_mode(False):`).
In this repo: `modules/smplx_fit/model.py` (loaders) and `modules/smplx_fit/fitting.py`
(`fit_smplx`/`resolve_edit` wrappers). See [[lesson-local-install-only]].
