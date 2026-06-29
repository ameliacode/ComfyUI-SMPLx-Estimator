"""
Click-Pose wrapper for headless 2D keypoint inference.

Repo: https://github.com/IDEA-Research/Click-Pose
Confirmed against: models/clickpose/clickpose.py, models/clickpose/postprocesses.py,
                   main.py, config/clickpose.cfg.py

Key facts:
- Model is built via build_model_main(args), which reads modelname from config.
- Input must be a NestedTensor (util.misc.nested_tensor_from_tensor_list).
- Postprocessors dict key is 'bbox' (not 'keypoints').
- Postprocessor output: results[0]['keypoints'] shape [num_select, 51]
    flattened as [x0,y0,v0, x1,y1,v1, ..., x16,y16,v16] in absolute pixel coords.
- results[0]['scores'] shape [num_select]: person-detection sigmoid scores.
"""

import argparse
import hashlib
import os
import sys
from typing import Optional

import numpy as np
import torch

# argparse.Namespace is stored in Click-Pose checkpoints; allow it for weights_only=True loads.
# Must be done at import time so it takes effect before any monkey-patches intercept torch.load.
if hasattr(torch.serialization, "add_safe_globals"):
    torch.serialization.add_safe_globals([argparse.Namespace])

VENDOR_DIR = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "vendor"
)
CLICKPOSE_DIR = os.path.join(VENDOR_DIR, "Click-Pose")
CLICKPOSE_CFG = os.path.join(CLICKPOSE_DIR, "config", "clickpose.cfg.py")

# ImageNet normalization (same as torchvision transforms.Normalize)
_PIXEL_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_PIXEL_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _ensure_in_path():
    if not os.path.isdir(CLICKPOSE_DIR):
        raise RuntimeError(
            f"Click-Pose repo not found at {CLICKPOSE_DIR}.\n"
            "Run: git clone https://github.com/IDEA-Research/Click-Pose vendor/Click-Pose"
        )
    if CLICKPOSE_DIR not in sys.path:
        sys.path.insert(0, CLICKPOSE_DIR)


class ClickPoseModel:
    """Wraps Click-Pose for headless single-image 2D keypoint detection."""

    NUM_JOINTS = 17

    def __init__(self):
        self.model = None
        self.postprocessors = None
        self.device: Optional[torch.device] = None
        # Cached from the most recent detect() call, used by refine()
        self._last_out: Optional[dict] = None
        self._last_encoder_memory: Optional[dict] = None
        self._last_image_id: Optional[str] = None

    def load(self, checkpoint_path: str, device: str = "cuda"):
        _ensure_in_path()

        from main import build_model_main, get_args_parser
        from util.config import Config

        parser = get_args_parser()
        # config_file is required=True in the parser, so pass it explicitly
        args = parser.parse_args(["--config_file", CLICKPOSE_CFG])
        args.device = device

        # Merge config file values into args (same as main.py does)
        cfg = Config.fromfile(args.config_file)
        cfg_dict = cfg._cfg_dict.to_dict()
        args_vars = vars(args)
        for k, v in cfg_dict.items():
            if k not in args_vars:
                setattr(args, k, v)

        self.device = torch.device(device)
        self.model, _, self.postprocessors = build_model_main(args)

        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        state_dict = checkpoint.get("model", checkpoint)
        self.model.load_state_dict(state_dict, strict=False)
        self.model.to(self.device)
        self.model.eval()

    @torch.no_grad()
    def detect(self, image_np: np.ndarray) -> dict:
        """
        Detect 2D keypoints from a single RGB image.

        Args:
            image_np: (H, W, 3) uint8 RGB numpy array

        Returns:
            dict:
                keypoints:  (17, 2) float32  absolute pixel coords [x, y]
                scores:     (17,)  float32  per-joint visibility score
                bbox_score: float            person detection confidence
                image_size: (H, W)
        """
        if self.model is None:
            raise RuntimeError("Call load() first.")

        _ensure_in_path()
        from util.misc import nested_tensor_from_tensor_list

        orig_h, orig_w = image_np.shape[:2]
        image_id = hashlib.md5(image_np.tobytes()).hexdigest()[:12]

        # Normalize to float, apply ImageNet stats → (3, H, W) tensor
        img = image_np.astype(np.float32) / 255.0
        img = (img - _PIXEL_MEAN) / _PIXEL_STD
        tensor = torch.from_numpy(img).permute(2, 0, 1).to(self.device)  # (3, H, W)

        # Click-Pose requires NestedTensor input
        samples = nested_tensor_from_tensor_list([tensor])

        # prepare_for_dn2 requires targets even in eval mode (to read device/batch-size).
        # Pass a minimal dummy target — only 'boxes' is accessed in the eval branch.
        dummy_targets = [{"boxes": torch.zeros((0, 4), device=self.device)}]

        # Capture encoder_memeory via forward hook so refine() can reuse it.
        # In non-refinement mode the transformer returns a 6-tuple; element [5] is the dict.
        _saved: dict = {}
        def _encoder_hook(module, inp, output):
            if isinstance(output, (tuple, list)) and len(output) == 6:
                _saved["encoder_memeory"] = output[5]
        handle = self.model.transformer.register_forward_hook(_encoder_hook)
        try:
            outputs = self.model(samples, targets=dummy_targets)
        finally:
            handle.remove()

        self._last_out = outputs
        self._last_encoder_memory = _saved.get("encoder_memeory")
        self._last_image_id = image_id

        # Postprocessors['bbox'] converts normalized coords → absolute pixel coords
        target_sizes = torch.tensor([[orig_h, orig_w]], device=self.device)
        results = self.postprocessors["bbox"](outputs, target_sizes)

        # results[0]['scores']:    [num_select]      person detection scores
        # results[0]['keypoints']: [num_select, 51]  x0,y0,v0,...,x16,y16,v16
        scores = results[0]["scores"].cpu().numpy()  # (num_select,)
        kps_flat = results[0]["keypoints"].cpu().numpy()  # (num_select, 51)

        if len(scores) == 0:
            return {
                "keypoints": np.zeros((self.NUM_JOINTS, 2), dtype=np.float32),
                "scores": np.zeros(self.NUM_JOINTS, dtype=np.float32),
                "bbox_score": 0.0,
                "image_size": (orig_h, orig_w),
                "image_id": image_id,
            }

        # Pick highest-confidence person detection
        best = int(np.argmax(scores))
        kps = kps_flat[best].reshape(self.NUM_JOINTS, 3)  # (17, 3): x, y, visibility

        return {
            "keypoints": kps[:, :2].astype(np.float32),  # (17, 2)
            "scores": kps[:, 2].astype(np.float32),  # (17,)
            "bbox_score": float(scores[best]),
            "image_size": (orig_h, orig_w),
            "image_id": image_id,
        }

    def get_refine_state(self) -> dict:
        """Snapshot the encoder/decoder state captured by the most recent detect().

        The caller captures this immediately after detect() and owns it, so the
        state survives later detect() calls on this shared, cached model instance
        (ComfyUI reuses one model across images and queue runs). Without this,
        a second image's detection would clobber the first's encoder memory and
        refine() would silently fall back to dumb coordinate snapping.
        """
        return {
            "out": self._last_out,
            "encoder_memory": self._last_encoder_memory,
            "image_id": self._last_image_id,
        }

    @torch.no_grad()
    def refine(self, refine_state: dict, user_corrections: dict, image_size: tuple) -> dict:
        """
        Run a single ClickPose refinement decoder pass with user-specified joint corrections.

        The decoder re-runs using the encoder memory captured by detect() (passed in
        via refine_state, NOT read from self, so it is immune to the shared model being
        re-run on another image). Corrected joints are frozen (delta_mask=0); all other
        joints are refined by the model.

        Args:
            refine_state:      dict from get_refine_state() captured right after detect()
            user_corrections:  {str(joint_idx): [px, py]}  absolute pixel coords
            image_size:        (H, W) of the original image

        Returns:
            Same dict format as detect(): {keypoints, scores, bbox_score, image_size}
        """
        out             = refine_state.get("out") if refine_state else None
        encoder_memeory = refine_state.get("encoder_memory") if refine_state else None
        state_image_id  = refine_state.get("image_id") if refine_state else None
        if out is None or encoder_memeory is None:
            raise RuntimeError("refine() requires detect() state (out/encoder_memory missing).")

        _ensure_in_path()
        from util.misc import inverse_sigmoid
        from util.keypoint_ops import keypoint_xyzxyz_to_xyxyzz

        H, W = image_size
        num_group_refine = self.model.num_group_refine   # 20
        num_body_points  = self.model.num_body_points    # 17
        nheads           = self.model.nheads             # 8
        group_bbox_kpt   = num_body_points + 1           # 18

        # `out` and `encoder_memeory` were captured at detect() time (see above).

        # ── Select top-K proposals by classification score ─────────────────────
        pred_cls = out["pred_logits"].clone()                       # (1, N, 2)
        pred_box = out["pred_boxes"].clone()                        # (1, N, 4)
        pred_kps = out["pred_keypoints"].clone()                    # (1, N, 51) xyxyzz normalized

        topk_idx = torch.topk(pred_cls.max(-1)[0], num_group_refine, dim=1)[1]  # (1, K)
        exp4  = topk_idx.unsqueeze(-1).expand(-1, -1, 4)
        exp2  = topk_idx.unsqueeze(-1).expand(-1, -1, 2)
        exp51 = topk_idx.unsqueeze(-1).expand(-1, -1, num_body_points * 3)

        boxes   = torch.gather(pred_box, 1, exp4)    # (1, K, 4)
        classes = torch.gather(pred_cls, 1, exp2)    # (1, K, 2)
        kps_51  = torch.gather(pred_kps, 1, exp51)   # (1, K, 51)

        # ── Build input_query_keypoint (1, K, 17, 2) ──────────────────────────
        # pred_keypoints is xyxyzz (interleaved): [x0,y0, x1,y1,...,x16,y16, v0,...,v16]
        # .view(K, 17, 2) gives [(x0,y0), (x1,y1), ..., (x16,y16)]  — matches NOC test code
        kps_2d = kps_51[:, :, :num_body_points * 2].clone().view(
            1, num_group_refine, num_body_points, 2
        )                                                                    # (1, K, 17, 2)

        # ── delta_mask: 1=model refines, 0=frozen user correction ─────────────
        delta_mask = torch.ones(1, num_group_refine, num_body_points, device=self.device)

        for joint_key, xy in user_corrections.items():
            j = int(joint_key)
            if 0 <= j < num_body_points:
                kps_2d[:, :, j, 0] = float(xy[0]) / W
                kps_2d[:, :, j, 1] = float(xy[1]) / H
                delta_mask[:, :, j] = 0.0

        # ── Attention mask: each group attends only to its own tokens ──────────
        total_q    = num_group_refine * group_bbox_kpt                       # 360
        attn_mask2 = torch.zeros(1, nheads, total_q, total_q,
                                  device=self.device, dtype=torch.bool)
        for q in range(total_q):
            sj = (q // group_bbox_kpt) * group_bbox_kpt
            ej = (q // group_bbox_kpt + 1) * group_bbox_kpt
            if sj > 0:
                attn_mask2[:, :, q, :sj] = True
            if ej < total_q:
                attn_mask2[:, :, q, ej:] = True
        attn_mask2 = attn_mask2.flatten(0, 1)                               # (bs*nheads, Tq, Tq)

        # ── Query encodings ───────────────────────────────────────────────────
        input_label_bbox       = torch.ones(1, num_group_refine, device=self.device, dtype=torch.long)
        input_query_bbox       = inverse_sigmoid(boxes)                      # (1, K, 4)
        input_query_label      = self.model.label_enc(input_label_bbox)     # (1, K, d_model)

        kpt_labels             = torch.arange(1, num_body_points + 1,
                                               device=self.device, dtype=torch.long)
        kpt_labels             = kpt_labels[None, None].expand(1, num_group_refine, -1)  # (1, K, 17)
        input_query_label_pose = self.model.label_enc_pose(kpt_labels)      # (1, K, 17, d_model)

        # ── Transformer refinement pass ───────────────────────────────────────
        # When refinement is not None, transformer.forward() skips num_box_decoder_layers
        # and returns (hs, references) with len(hs) = dec_layers - num_box_decoder_layers = 4
        hs, reference = self.model.transformer(
            None, None,
            input_query_bbox,        # refpoint_embed   (1, K, 4)
            None,
            input_query_label,       # tgt              (1, K, d_model)
            kps_2d,                  # refpoint_embed_pose (1, K, 17, 2)
            input_query_label_pose,  # tgt_pose         (1, K, 17, d_model)
            None,
            attn_mask2,              # attn_mask2
            delta_mask,              # refinement       (1, K, 17)
            encoder_memeory,
        )

        # ── Decode keypoints from last refinement decoder layer ───────────────
        kpt_index = [q for q in range(total_q) if q % group_bbox_kpt != 0]
        kpt_idx_t = torch.tensor(kpt_index, device=self.device)

        # hs: list of (bs, K*group_bbox_kpt, d_model);  reference: n+1 entries
        dec_lid   = len(hs) - 1
        layer_hs  = hs[dec_lid]                                              # (1, total_q, d)
        layer_ref = reference[dec_lid]                                       # (1, total_q, 4)

        layer_hs_kpt  = layer_hs.index_select(1, kpt_idx_t)                 # (1, K*17, d)
        layer_ref_kpt = layer_ref.index_select(1, kpt_idx_t)                # (1, K*17, 4)

        delta_xy_unsig = self.model.pose_embed[dec_lid](layer_hs_kpt)       # (1, K*17, 2)

        kpt_out = (
            delta_xy_unsig * delta_mask.view(1, -1, 1)
            + inverse_sigmoid(layer_ref_kpt[..., :2])
        )
        vis = torch.ones_like(kpt_out)
        xyv = torch.cat((kpt_out, vis[:, :, :1]), dim=-1).sigmoid()         # (1, K*17, 3)
        layer_res = xyv.reshape(1, num_group_refine, num_body_points, 3).flatten(2, 3)  # (1, K, 51)
        layer_res = keypoint_xyzxyz_to_xyxyzz(layer_res)                    # xyxyzz format

        # ── Pick best person (highest person-detection class score) ───────────
        cls_probs = classes.sigmoid().max(-1)[0]                             # (1, K)
        best      = int(cls_probs[0].argmax())

        # layer_res is xyxyzz (interleaved): [x0,y0, x1,y1,...,x16,y16, v0,...,v16]
        kps_raw = layer_res[0, best].cpu().numpy()                           # (51,)
        kps_xy  = kps_raw[:num_body_points * 2].reshape(num_body_points, 2) # (17,2) normalized
        v_score = kps_raw[num_body_points * 2:]                              # (17,)

        kps_abs = (kps_xy * np.array([[W, H]], dtype=np.float32)).astype(np.float32)

        return {
            "keypoints":  kps_abs,                       # (17, 2) absolute pixel coords
            "scores":     v_score.astype(np.float32),    # (17,)
            "bbox_score": float(cls_probs[0, best]),
            "image_size": (H, W),
            "image_id":   state_image_id,
        }
