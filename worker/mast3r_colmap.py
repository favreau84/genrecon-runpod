"""MASt3R-SfM -> COLMAP texte (cameras/images/points3D.txt) pour GenRecon mode Iphone.

Reprend le chemin du demo officiel (make_pairs + sparse_global_alignment) avec
le checkpoint métrique, puis écrit directement le format texte COLMAP attendu
par le chunker/selecter GenRecon (PINHOLE, poses w2c, points sans track).
Le monde est aligné gravité (moyenne des vecteurs "haut" caméra -> +Z), comme
les scènes ScanNet++/MASt3R sur lesquelles GenRecon a été mis au point.
"""

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from scipy.spatial.transform import Rotation

from mast3r.model import AsymmetricMASt3R
from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
import mast3r.utils.path_to_dust3r  # noqa: F401 — ajoute dust3r au sys.path
from dust3r.image_pairs import make_pairs
from dust3r.utils.image import load_images

MODEL = os.environ.get("MAST3R_MODEL", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric")
MAX_POINTS = int(os.environ.get("MAST3R_MAX_POINTS", "300000"))
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


def gravity_align(cam2w, pts_list):
    """Tourne le monde pour amener la moyenne des 'haut caméra' sur +Z."""
    up = -cam2w[:, :3, 1].mean(axis=0)  # haut caméra en monde = -(2e colonne de R c2w)
    up /= np.linalg.norm(up)
    fwd0 = cam2w[0, :3, 2]
    x_axis = fwd0 - np.dot(fwd0, up) * up
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(up, x_axis)
    A = np.stack([x_axis, y_axis, up])  # X' = A @ X
    print(f"alignement gravité : up estimé {np.round(up, 3).tolist()}", flush=True)
    T = np.eye(4)
    T[:3, :3] = A
    cam2w = T @ cam2w
    pts_list = [(A @ p.T).T for p in pts_list]
    return cam2w, pts_list


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scene_dir", required=True)
    args = ap.parse_args()
    scene_dir = Path(args.scene_dir)
    image_dir = scene_dir / "images"
    paths = sorted(str(p) for p in image_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS)
    if len(paths) < 2:
        raise ValueError(f"Pas assez d'images dans {image_dir}")

    model = AsymmetricMASt3R.from_pretrained(MODEL).to("cuda").eval()
    print("MASt3R chargé", flush=True)

    imgs = load_images(paths, size=512, verbose=False)
    graph = os.environ.get("MAST3R_SCENE_GRAPH", "logwin-8")
    pairs = make_pairs(imgs, scene_graph=graph, prefilter=None, symmetrize=True)
    print(f"{len(paths)} images, {len(pairs)} paires ({graph})", flush=True)

    cache = "/tmp/mast3r_cache"
    shutil.rmtree(cache, ignore_errors=True)
    scene = sparse_global_alignment(
        paths, pairs, cache, model,
        lr1=0.07, niter1=500, lr2=0.014, niter2=200,
        opt_depth=True, matching_conf_thr=5.0, shared_intrinsics=True,
        device="cuda",
    )

    cam2w = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)  # (N,4,4)
    Ks = [k.detach().cpu().numpy() for k in scene.intrinsics]  # K des images redimensionnées
    pts_list = [p.detach().cpu().numpy().astype(np.float64) for p in scene.get_sparse_pts3d()]
    col_list = [np.asarray(c) for c in scene.pts3d_colors]  # RGB [0,1]

    cam2w, pts_list = gravity_align(cam2w, pts_list)

    # Sanité échelle (checkpoint métrique -> mètres attendus)
    med = np.median([np.median(np.linalg.norm(p - c[None, :3, 3], axis=1))
                     for p, c in zip(pts_list, cam2w) if len(p)])
    print(f"distance caméra->points médiane : {med:.2f} m", flush=True)

    colmap_dir = scene_dir / "colmap"
    colmap_dir.mkdir(parents=True, exist_ok=True)

    # cameras.txt + images.txt — intrinsèques remises à l'échelle des fichiers originaux
    cam_lines, img_lines = [], []
    for i, p in enumerate(paths):
        W0, H0 = Image.open(p).size
        H1, W1 = (int(v) for v in np.asarray(imgs[i]["true_shape"]).flatten())
        sx, sy = W0 / W1, H0 / H1
        K = Ks[i]
        cam_lines.append(
            f"{i + 1} PINHOLE {W0} {H0} {K[0, 0] * sx:.6f} {K[1, 1] * sy:.6f} {K[0, 2] * sx:.6f} {K[1, 2] * sy:.6f}"
        )
        w2c = np.linalg.inv(cam2w[i])
        q = Rotation.from_matrix(w2c[:3, :3]).as_quat()  # [x, y, z, w]
        t = w2c[:3, 3]
        img_lines.append(
            f"{i + 1} {q[3]:.9f} {q[0]:.9f} {q[1]:.9f} {q[2]:.9f} "
            f"{t[0]:.9f} {t[1]:.9f} {t[2]:.9f} {i + 1} {Path(p).name}"
        )
        img_lines.append("")  # ligne POINTS2D vide (pas de tracks)

    pts = np.concatenate([p for p in pts_list if len(p)])
    cols = (np.concatenate([c for c in col_list if len(c)]).clip(0, 1) * 255).astype(np.uint8)
    finite = np.isfinite(pts).all(axis=1)
    pts, cols = pts[finite], cols[finite]
    if len(pts) > MAX_POINTS:
        sel = np.random.default_rng(42).choice(len(pts), MAX_POINTS, replace=False)
        pts, cols = pts[sel], cols[sel]

    (colmap_dir / "cameras.txt").write_text(
        "# Camera list: CAMERA_ID MODEL WIDTH HEIGHT PARAMS[]\n" + "\n".join(cam_lines) + "\n"
    )
    (colmap_dir / "images.txt").write_text(
        "# Image list: IMAGE_ID QW QX QY QZ TX TY TZ CAMERA_ID NAME\n" + "\n".join(img_lines) + "\n"
    )
    with (colmap_dir / "points3D.txt").open("w") as f:
        f.write("# 3D point list: POINT3D_ID X Y Z R G B ERROR TRACK[]\n")
        for j, (p, c) in enumerate(zip(pts, cols)):
            f.write(f"{j + 1} {p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {c[0]} {c[1]} {c[2]} 0.0\n")

    print(f"COLMAP texte écrit : {len(paths)} images, {len(pts)} points -> {colmap_dir}", flush=True)


if __name__ == "__main__":
    # pas de torch.no_grad() global : sparse_global_alignment optimise par gradient
    main()
