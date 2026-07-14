"""Inférence headless Plane-DUSt3R : images d'une pièce -> plan_raw.json.

Remplace custom.py (qui ouvre une fenêtre Open3D et n'écrit rien) : même
pipeline dust3r_extract -> extract_plane -> plane_merge, mais sérialise
node_data + centres caméra + gestion de l'échelle métrique.

Échelle :
  - metric_flag=True (scene.preset_metric() du fork) donne l'échelle "brute"
    du réseau DUSt3R, approximativement métrique mais non garantie ;
  - sanity check : hauteur caméra au-dessus du sol dans [1.0, 2.0] m (iPhone à
    la main) et diamètre de pièce dans [1.5, 20] m ;
  - si le check échoue en scale_mode=auto : fallback via le MASt3R métrique
    vendoré (sparse_global_alignment sur les mêmes images), facteur d'échelle
    de type Umeyama entre les deux nuages de centres caméra.

À lancer avec cwd=/opt/planedust3r (cfg NonCuboidRoom/cfg.yaml relative).
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")  # plane_merge importe matplotlib

REPO = Path(os.environ.get("PLANEDUST3R_DIR", "/opt/planedust3r"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "MASt3R"))  # mast3r + (via path_to_dust3r) dust3r
sys.path.append(str(Path(__file__).parent / "vendor"))  # stub mmcv pour hrnet

import numpy as np  # noqa: E402
import torch  # noqa: E402
import yaml  # noqa: E402
from easydict import EasyDict  # noqa: E402
from PIL import Image  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}

CAM_HEIGHT_RANGE = (1.0, 2.0)   # m, iPhone tenu à la main
ROOM_DIAMETER_RANGE = (1.5, 20.0)  # m


def _load_models(dust3r_ckpt, noncuboid_ckpt, device):
    from dust3r.model import AsymmetricCroCo3DStereo
    from NonCuboidRoom.noncuboid.models import Detector

    dust3r_model = AsymmetricCroCo3DStereo.from_pretrained(dust3r_ckpt).to(device)
    noncuboid_model = Detector()
    state_dict = torch.load(noncuboid_ckpt, map_location=torch.device(device))
    noncuboid_model.load_state_dict(state_dict)
    with open(REPO / "NonCuboidRoom" / "cfg.yaml") as f:
        cfg = EasyDict(yaml.safe_load(f))
    return dust3r_model, noncuboid_model, cfg


def _metric_check(node_data, cam_centers, scale=1.0):
    """Hauteur caméra / diamètre de pièce plausibles après mise à l'échelle ?"""
    result = {"scale_tested": scale, "cam_height_m": None, "room_diameter_m": None, "passed": False}
    floor = node_data.get("floor_pparam") or node_data.get("ceiling_pparam")
    if not floor:
        return result
    n = np.array(floor[:3], float)
    n /= np.linalg.norm(n)
    d = float(floor[3]) * scale
    cams = np.array(cam_centers, float) * scale
    heights = np.abs(cams @ n + d)
    result["cam_height_m"] = round(float(np.mean(heights)), 3)

    endpoints = [
        np.array(p, float) * scale
        for w in node_data["global_plane_info"]
        for p in (w.get("left_endpoint"), w.get("right_endpoint"))
        if p is not None
    ]
    if len(endpoints) >= 2:
        pts = np.stack(endpoints)
        result["room_diameter_m"] = round(float(np.linalg.norm(pts.max(0) - pts.min(0))), 3)

    ok_h = CAM_HEIGHT_RANGE[0] <= result["cam_height_m"] <= CAM_HEIGHT_RANGE[1]
    ok_d = result["room_diameter_m"] is None or (
        ROOM_DIAMETER_RANGE[0] <= result["room_diameter_m"] <= ROOM_DIAMETER_RANGE[1]
    )
    result["passed"] = bool(ok_h and ok_d)
    return result


def _detect_openings(image_list, dust3r_output, device):
    """Détection zero-shot portes/fenêtres (OWLv2) par frame, puis extraction
    des points 3D du pointmap DUSt3R dans chaque box. L'assignation aux murs et
    la fusion multi-vues se font dans layout_to_geojson (rejouable en local)."""
    from transformers import pipeline

    model_name = os.environ.get("OPENINGS_MODEL", "google/owlv2-base-patch16-ensemble")
    threshold = float(os.environ.get("OPENINGS_THRESHOLD", "0.25"))
    detector = pipeline(
        "zero-shot-object-detection", model=model_name,
        device=0 if device == "cuda" else -1,
    )
    rng = np.random.default_rng(42)
    out = []
    for img_id, path in enumerate(image_list):
        img = Image.open(path).convert("RGB")
        W, H = img.size
        preds = detector(img, candidate_labels=["a door", "a window"], threshold=threshold)
        pts3d = np.asarray(dust3r_output["pts3d"][img_id])   # (h, w, 3)
        conf = np.asarray(dust3r_output["confidence"][img_id])  # masque (h, w)
        gh, gw = pts3d.shape[:2]
        for p in preds:
            box = p["box"]
            x0 = max(int(box["xmin"] / W * gw), 0)
            x1 = min(int(np.ceil(box["xmax"] / W * gw)), gw)
            y0 = max(int(box["ymin"] / H * gh), 0)
            y1 = min(int(np.ceil(box["ymax"] / H * gh)), gh)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            patch = pts3d[y0:y1, x0:x1].reshape(-1, 3)
            mask = conf[y0:y1, x0:x1].reshape(-1).astype(bool)
            pts = patch[mask] if mask.any() else patch
            pts = pts[np.isfinite(pts).all(axis=1)]
            if len(pts) < 8:
                continue
            if len(pts) > 200:
                pts = pts[rng.choice(len(pts), 200, replace=False)]
            out.append({
                "label": "door" if "door" in p["label"] else "window",
                "score": round(float(p["score"]), 3),
                "img_id": img_id,
                "points": np.round(pts, 4).tolist(),
            })
    print(f"[infer] ouvertures : {len(out)} détection(s) brute(s) sur {len(image_list)} frames",
          flush=True)
    return out


def _umeyama_scale(image_list, dust3r_centers, device):
    """Facteur d'échelle entre les centres caméra DUSt3R et ceux du MASt3R
    métrique (mêmes images, même ordre) : ratio des dispersions RMS autour du
    centroïde (la composante échelle d'Umeyama — rotation/translation inutiles)."""
    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.image_pairs import make_pairs
    from dust3r.utils.image import load_images

    model_name = os.environ.get(
        "MAST3R_MODEL", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    model = AsymmetricMASt3R.from_pretrained(model_name).to(device).eval()
    imgs = load_images(image_list, size=512, verbose=False)
    pairs = make_pairs(imgs, scene_graph="complete", prefilter=None, symmetrize=True)
    cache = "/tmp/mast3r_scale_cache"
    shutil.rmtree(cache, ignore_errors=True)
    try:
        scene = sparse_global_alignment(
            image_list, pairs, cache, model,
            lr1=0.07, niter1=500, lr2=0.014, niter2=200,
            opt_depth=True, matching_conf_thr=5.0, shared_intrinsics=True,
            device=device,
        )
    except TypeError:
        # le fork vendoré peut différer du mast3r officiel sur les kwargs
        scene = sparse_global_alignment(image_list, pairs, cache, model, device=device)
    metric_centers = scene.get_im_poses().detach().cpu().numpy()[:, :3, 3]

    a = np.asarray(dust3r_centers, float)
    b = np.asarray(metric_centers, float)
    if len(a) != len(b) or len(a) < 3:
        raise RuntimeError(f"centres caméra incomparables ({len(a)} vs {len(b)})")
    da = a - a.mean(axis=0)
    db = b - b.mean(axis=0)
    denom = float(np.sqrt((da ** 2).sum()))
    if denom < 1e-6:
        raise RuntimeError("caméras DUSt3R quasi confondues — échelle indéterminable")
    return float(np.sqrt((db ** 2).sum()) / denom)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--scale_mode", default="auto", choices=["auto", "metric", "none"])
    ap.add_argument("--dust3r_ckpt",
                    default=os.environ.get("PLANEDUST3R_CKPT",
                                           "/runpod-volume/planedust3r/checkpoint-best-onlyencoder.pth"))
    ap.add_argument("--noncuboid_ckpt",
                    default=os.environ.get("NONCUBOID_CKPT",
                                           "/runpod-volume/planedust3r/Structured3D_pretrained.pt"))
    ap.add_argument("--threshold", type=float, nargs=4, default=[0.35, 0.25, 0.25, 0.3])
    ap.add_argument("--detect_openings", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from MASt3R.dust3r_extract import dust3r_extract
    from NonCuboidRoom.plane_detection import extract_plane
    from plane_merge_planedust3r import plane_merge

    image_list = sorted(
        str(p) for p in Path(args.image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if len(image_list) < 2:
        raise ValueError(f"Pas assez d'images dans {args.image_dir} ({len(image_list)})")
    src_size = Image.open(image_list[0]).size
    # Dust3rDataset (NonCuboid) redimensionne TOUTE image en 1280x720 : les
    # détections vivent dans cet espace, pas dans celui de l'image d'origine.
    image_size = (1280, 720)
    print(f"[infer] {len(image_list)} images {src_size[0]}x{src_size[1]} "
          f"(détections en {image_size[0]}x{image_size[1]})", flush=True)

    dust3r_model, noncuboid_model, cfg = _load_models(
        args.dust3r_ckpt, args.noncuboid_ckpt, args.device
    )
    print("[infer] modèles chargés", flush=True)

    metric_flag = args.scale_mode in ("auto", "metric")
    dust3r_output = dust3r_extract(image_list, dust3r_model, device=args.device, metric=metric_flag)
    dust3r_image_size = (dust3r_output["pts3d"][0].shape[1], dust3r_output["pts3d"][0].shape[0])
    print(f"[infer] dust3r ok (pointmaps {dust3r_image_size})", flush=True)

    plane_detection = extract_plane(
        image_list, noncuboid_model, cfg, threshold=tuple(args.threshold)
    )
    print("[infer] détection de plans 2D ok", flush=True)

    node_data = plane_merge(
        dust3r_output, plane_detection,
        metric=metric_flag, image_size=image_size, dust3r_image_size=dust3r_image_size,
    )
    n_walls = len(node_data["global_plane_info"])
    print(f"[infer] fusion : {n_walls} murs, sol={bool(node_data['floor_pparam'])}, "
          f"plafond={bool(node_data['ceiling_pparam'])}", flush=True)

    openings_raw = []
    if args.detect_openings:
        try:
            openings_raw = _detect_openings(image_list, dust3r_output, args.device)
        except Exception as e:  # les ouvertures sont un bonus, pas un bloqueur
            print(f"[infer] détection d'ouvertures échouée : {e}", flush=True)

    cam_centers = np.asarray(dust3r_output["poses"])[:, :3, 3].tolist()

    scale_factor = 1.0
    scale_mode_out = "metric" if metric_flag else "none"
    check = _metric_check(node_data, cam_centers)
    print(f"[infer] sanity échelle (brute) : {check}", flush=True)
    if not check["passed"] and args.scale_mode == "auto":
        print("[infer] échelle DUSt3R non plausible → fallback MASt3R métrique (Umeyama)", flush=True)
        try:
            scale_factor = _umeyama_scale(image_list, cam_centers, args.device)
            scale_mode_out = "umeyama"
            check = _metric_check(node_data, cam_centers, scale=scale_factor)
            print(f"[infer] facteur {scale_factor:.4f} — re-check : {check}", flush=True)
        except Exception as e:  # le plan reste exploitable, l'échelle est marquée douteuse
            print(f"[infer] fallback Umeyama échoué : {e}", flush=True)
            scale_mode_out = "metric-unverified"

    raw = dict(node_data)
    raw.update({
        "openings_raw": openings_raw,
        "cam_centers": cam_centers,
        "image_names": [Path(p).name for p in image_list],
        "scale_mode": scale_mode_out,
        "scale_factor": scale_factor,
        "metric_check": check,
    })
    with open(args.out_json, "w") as f:
        json.dump(raw, f)
    print(f"[infer] plan_raw écrit -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
