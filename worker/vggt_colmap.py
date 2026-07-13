"""VGGT feedforward -> reconstruction COLMAP (sparse/, format binaire).

Copie du chemin no-BA de vggt/demo_colmap.py (commit a288dd0), sans la branche
bundle-adjustment ni ses imports : le top-level de demo_colmap.py importe
vggt.dependency.track_predict -> vggsfm_utils -> lightglue, absent de l'image
et inutile en mode feedforward.
"""

import argparse
import copy
import glob
import os
import random

import numpy as np
import torch
import torch.nn.functional as F
import trimesh

from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images_square
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map
from vggt.utils.helper import create_pixel_coordinate_grid, randomly_limit_trues
from vggt.dependency.np_to_pycolmap import batch_np_matrix_to_pycolmap_wo_track


def parse_args():
    parser = argparse.ArgumentParser(description="VGGT -> COLMAP (feedforward)")
    parser.add_argument("--scene_dir", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--conf_thres_value", type=float, default=5.0)
    return parser.parse_args()


def run_VGGT(model, images, dtype, resolution=518):
    assert len(images.shape) == 4 and images.shape[1] == 3

    images = F.interpolate(images, size=(resolution, resolution), mode="bilinear", align_corners=False)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    return (
        extrinsic.squeeze(0).cpu().numpy(),
        intrinsic.squeeze(0).cpu().numpy(),
        depth_map.squeeze(0).cpu().numpy(),
        depth_conf.squeeze(0).cpu().numpy(),
    )


def rename_colmap_recons_and_rescale_camera(
    reconstruction, image_paths, original_coords, img_size, shift_point2d_to_original_res=False, shared_camera=False
):
    rescale_camera = True

    for pyimageid in reconstruction.images:
        pyimage = reconstruction.images[pyimageid]
        pycamera = reconstruction.cameras[pyimage.camera_id]
        pyimage.name = image_paths[pyimageid - 1]

        if rescale_camera:
            pred_params = copy.deepcopy(pycamera.params)

            real_image_size = original_coords[pyimageid - 1, -2:]
            resize_ratio = max(real_image_size) / img_size
            pred_params = pred_params * resize_ratio
            real_pp = real_image_size / 2
            pred_params[-2:] = real_pp

            pycamera.params = pred_params
            pycamera.width = real_image_size[0]
            pycamera.height = real_image_size[1]

        if shift_point2d_to_original_res:
            top_left = original_coords[pyimageid - 1, :2]
            for point2D in pyimage.points2D:
                point2D.xy = (point2D.xy - top_left) * resize_ratio

        if shared_camera:
            rescale_camera = False

    return reconstruction


def main(args):
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    device = "cuda"

    model = VGGT()
    _URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
    model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
    model.eval()
    model = model.to(device)
    print("Model loaded", flush=True)

    image_dir = os.path.join(args.scene_dir, "images")
    image_path_list = sorted(glob.glob(os.path.join(image_dir, "*")))
    if len(image_path_list) == 0:
        raise ValueError(f"No images found in {image_dir}")
    base_image_path_list = [os.path.basename(path) for path in image_path_list]

    vggt_fixed_resolution = 518
    img_load_resolution = 1024

    images, original_coords = load_and_preprocess_images_square(image_path_list, img_load_resolution)
    images = images.to(device)
    original_coords = original_coords.to(device)
    print(f"Loaded {len(images)} images from {image_dir}", flush=True)

    extrinsic, intrinsic, depth_map, depth_conf = run_VGGT(model, images, dtype, vggt_fixed_resolution)

    # Normalisation d'échelle : VGGT est à échelle arbitraire, mais le chunker
    # GenRecon (radius_m, tailles de chunks) suppose des mètres. On recale la
    # profondeur médiane des pixels confiants sur une distance de prise de vue
    # typique d'intérieur (SCENE_MEDIAN_DEPTH_M, défaut 2.5 m).
    target_med = float(os.environ.get("SCENE_MEDIAN_DEPTH_M", "2.5"))
    conf_med_mask = depth_conf >= max(1.05, float(np.quantile(depth_conf, 0.5)))
    med_depth = float(np.median(depth_map[..., 0][conf_med_mask] if depth_map.ndim == 4 else depth_map[conf_med_mask]))
    scale = target_med / max(med_depth, 1e-6)
    print(f"profondeur médiane VGGT: {med_depth:.3f} -> échelle x{scale:.3f} (cible {target_med} m)", flush=True)
    depth_map = depth_map * scale
    extrinsic = extrinsic.copy()
    extrinsic[:, :3, 3] *= scale

    points_3d = unproject_depth_map_to_point_map(depth_map, extrinsic, intrinsic)

    # Chemin feedforward (sans BA) de demo_colmap.py
    conf_thres_value = args.conf_thres_value
    max_points_for_colmap = 100000
    shared_camera = False
    camera_type = "PINHOLE"

    image_size = np.array([vggt_fixed_resolution, vggt_fixed_resolution])
    num_frames, height, width, _ = points_3d.shape

    points_rgb = F.interpolate(
        images, size=(vggt_fixed_resolution, vggt_fixed_resolution), mode="bilinear", align_corners=False
    )
    points_rgb = (points_rgb.cpu().numpy() * 255).astype(np.uint8)
    points_rgb = points_rgb.transpose(0, 2, 3, 1)

    points_xyf = create_pixel_coordinate_grid(num_frames, height, width)

    # Seuil adaptatif : le défaut demo (5.0) peut éliminer ~tous les points
    # selon la scène (observé : 0 point -> chunker GenRecon plante). On garde
    # au minimum les 25 % de pixels les plus confiants, plancher à 1.05.
    q = [float(np.quantile(depth_conf, p)) for p in (0.25, 0.5, 0.75, 0.9)]
    print(f"depth_conf quantiles 25/50/75/90: {[round(v, 2) for v in q]}", flush=True)
    conf_thres = min(conf_thres_value, max(1.05, q[2]))
    if conf_thres != conf_thres_value:
        print(f"conf_thres_value {conf_thres_value} -> {conf_thres:.2f} (adaptatif)", flush=True)

    conf_mask = depth_conf >= conf_thres
    print(f"points au-dessus du seuil : {int(conf_mask.sum())}", flush=True)
    conf_mask = randomly_limit_trues(conf_mask, max_points_for_colmap)

    points_3d = points_3d[conf_mask]
    points_xyf = points_xyf[conf_mask]
    points_rgb = points_rgb[conf_mask]

    print("Converting to COLMAP format", flush=True)
    reconstruction = batch_np_matrix_to_pycolmap_wo_track(
        points_3d,
        points_xyf,
        points_rgb,
        extrinsic,
        intrinsic,
        image_size,
        shared_camera=shared_camera,
        camera_type=camera_type,
    )

    reconstruction = rename_colmap_recons_and_rescale_camera(
        reconstruction,
        base_image_path_list,
        original_coords.cpu().numpy(),
        img_size=vggt_fixed_resolution,
        shift_point2d_to_original_res=True,
        shared_camera=shared_camera,
    )

    sparse_dir = os.path.join(args.scene_dir, "sparse")
    os.makedirs(sparse_dir, exist_ok=True)
    reconstruction.write(sparse_dir)
    trimesh.PointCloud(points_3d, colors=points_rgb).export(os.path.join(sparse_dir, "points.ply"))
    print(f"Saved reconstruction to {sparse_dir}", flush=True)


if __name__ == "__main__":
    with torch.no_grad():
        main(parse_args())
