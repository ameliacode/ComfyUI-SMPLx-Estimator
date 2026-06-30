"""
Export Mesh — write a MESH_DATA bundle (vertices + faces, e.g. from the SMPL-X
Editor) to a mesh file in ComfyUI's output directory and return its path.

OBJ / PLY are written directly (no deps); GLB uses trimesh if installed.
"""

import os

import numpy as np
import folder_paths


def _write_obj(path, verts, faces):
    with open(path, "w") as f:
        for v in verts:
            f.write(f"v {v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:                       # OBJ is 1-indexed
            f.write(f"f {t[0] + 1} {t[1] + 1} {t[2] + 1}\n")


def _write_ply(path, verts, faces):
    with open(path, "w") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {len(verts)}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write(f"element face {len(faces)}\n")
        f.write("property list uchar int vertex_indices\nend_header\n")
        for v in verts:
            f.write(f"{v[0]:.6f} {v[1]:.6f} {v[2]:.6f}\n")
        for t in faces:
            f.write(f"3 {t[0]} {t[1]} {t[2]}\n")


def _write_glb(path, verts, faces):
    try:
        import trimesh
    except ImportError:
        raise RuntimeError(
            "GLB export needs trimesh (pip install trimesh). Choose obj or ply instead."
        )
    trimesh.Trimesh(vertices=verts, faces=faces, process=False).export(path)


_WRITERS = {"obj": _write_obj, "ply": _write_ply, "glb": _write_glb}


class ExportMesh:
    """Write a MESH_DATA bundle to disk; outputs the saved file path."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mesh_data": ("MESH_DATA",),
                "filename_prefix": ("STRING", {"default": "smplx_mesh"}),
                "format": (["obj", "ply", "glb"],),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("file_path",)
    FUNCTION = "export"
    CATEGORY = "SMPLx Estimator"
    OUTPUT_NODE = True

    def export(self, mesh_data, filename_prefix, format):
        verts = np.asarray(mesh_data["vertices"], np.float32)
        faces = np.asarray(mesh_data["faces"], np.int64)
        out_dir = folder_paths.get_output_directory()
        full, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, out_dir)
        name = f"{filename}_{counter:05}.{format}"
        path = os.path.join(full, name)
        _WRITERS[format](path, verts, faces)
        print(f"[export_mesh] wrote {len(verts)} verts / {len(faces)} faces -> {path}")
        return {"ui": {"text": [path]}, "result": (path,)}
