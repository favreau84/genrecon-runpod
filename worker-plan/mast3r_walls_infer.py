"""Mode walls : photos ordonnées des murs (ordre horaire) -> plan_raw.json.

MASt3R-SfM sur un graphe de paires restreint dérivé de l'ORDRE de prise de vue
(consécutives à distance 1 et 2, cyclique = fermeture de boucle), puis toute la
géométrie (verticale, clustering yaw, plans de murs) est déléguée à
walls_geometry.py — rejouable en local sans GPU.

À lancer avec cwd quelconque (sys.path autonome). Sorties :
    --out_json    plan_raw.json  (schéma commun layout_to_geojson)
    --out_report  walls_report.json  (qualité par paire, clusters, fits)
    --out_debug   plan_debug.png  (rendu top-down)
"""

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

REPO = Path(os.environ.get("PLANEDUST3R_DIR", "/opt/planedust3r"))
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "MASt3R"))  # mast3r + (via path_to_dust3r) dust3r
sys.path.insert(0, str(Path(__file__).parent))

import numpy as np  # noqa: E402
import torch  # noqa: E402

import walls_geometry  # noqa: E402

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
CACHE = "/tmp/mast3r_walls_cache"
# Une paire est "faible" si peu de correspondances OU confiance moyenne de
# matching basse. (PAS le conf_score des pointmaps : systématiquement ~2-5 sur
# de l'ultra grand angle alors que le matching est excellent — calibré sur le
# run test-chambre-2 : 1700-6000 corres/paire, conf_mean 1.9-24.6.)
WEAK_N_CORRES = 1000
WEAK_CONF_MEAN = 2.0


def _build_pairs(imgs):
    """Paires séquentielles cycliques à distance 1 et 2 (dédoublonnées puis
    SYMÉTRISÉES — l'optimiseur de sparse_ga indexe is_matching_ok dans les
    deux sens, cf. make_pairs(symmetrize=True) dans mast3r_extract.py) :
    l'ordre de prise de vue donne le graphe — pas de matching N², et la
    distance 2 sert de chemin de secours si un coin est mal couvert."""
    n = len(imgs)
    pairs, seen = [], set()
    for k in (1, 2):
        for i in range(n):
            j = (i + k) % n
            key = (min(i, j), max(i, j))
            if i != j and key not in seen:
                seen.add(key)
                pairs.append((imgs[i], imgs[j]))
    pairs += [(b, a) for a, b in pairs]  # symétrisation
    return pairs


def _pair_scores(cache, image_list, pairs):
    """Qualité de matching par paire depuis le cache de sparse_ga :
    ((conf_score, conf_sum, n_corres), corres). Best-effort."""
    from mast3r.utils.misc import hash_md5

    cdir = Path(cache) / "corres_conf=desc_conf_subsample=8"
    out = []
    seen = set()
    for a, b in pairs:
        i, j = a["idx"], b["idx"]
        if (min(i, j), max(i, j)) in seen:  # paires symétrisées : un seul sens
            continue
        seen.add((min(i, j), max(i, j)))
        pi, pj = image_list[i], image_list[j]
        score = None
        for p1, p2 in ((pi, pj), (pj, pi)):
            f = cdir / f"{hash_md5(p1)}-{hash_md5(p2)}.pth"
            if f.exists():
                try:
                    (conf_score, conf_sum, n_corres), _ = torch.load(f, map_location="cpu")
                    score = {"conf": round(float(conf_score), 3),
                             "n_corres": int(n_corres),
                             "conf_mean": round(float(conf_sum) / max(int(n_corres), 1), 3)}
                except Exception:
                    pass
                break
        entry = {"i": i, "j": j,
                 "names": [Path(pi).name, Path(pj).name],
                 "consecutive": abs(i - j) in (1, len(image_list) - 1)}
        if score:
            entry.update(score)
            entry["weak"] = (score["n_corres"] < WEAK_N_CORRES
                             or score["conf_mean"] < WEAK_CONF_MEAN)
        else:
            entry["weak"] = None  # score indisponible
        out.append(entry)
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--image_dir", required=True)
    ap.add_argument("--out_json", required=True)
    ap.add_argument("--out_report", required=True)
    ap.add_argument("--out_debug", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    from mast3r.model import AsymmetricMASt3R
    from mast3r.cloud_opt.sparse_ga import sparse_global_alignment
    import mast3r.utils.path_to_dust3r  # noqa: F401
    from dust3r.utils.image import load_images

    image_list = sorted(
        str(p) for p in Path(args.image_dir).iterdir() if p.suffix.lower() in IMAGE_EXTS
    )
    if len(image_list) < 3:
        raise ValueError(f"au moins 3 photos requises ({len(image_list)} trouvées)")
    print(f"[walls] {len(image_list)} photos (ordre horaire = ordre des noms)", flush=True)

    model_name = os.environ.get(
        "MAST3R_MODEL", "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
    )
    model = AsymmetricMASt3R.from_pretrained(model_name).to(args.device).eval()
    print("[walls] MASt3R métrique chargé", flush=True)

    imgs = load_images(image_list, size=512, verbose=False)
    pairs = _build_pairs(imgs)
    print(f"[walls] graphe séquentiel cyclique : {len(pairs)} paires (distances 1 et 2)", flush=True)

    shutil.rmtree(CACHE, ignore_errors=True)
    try:
        scene = sparse_global_alignment(
            image_list, pairs, CACHE, model,
            lr1=0.07, niter1=500, lr2=0.014, niter2=200,
            opt_depth=True, matching_conf_thr=5.0, shared_intrinsics=True,  # même iPhone 0,5×
            device=args.device,
        )
    except AssertionError as e:
        # MST déconnecté : identifier les maillons faibles pour guider la reprise
        scores = _pair_scores(CACHE, image_list, pairs)
        weak = sorted((s for s in scores if s.get("n_corres") is not None),
                      key=lambda s: s["n_corres"])[:3]
        detail = ", ".join(f"{s['names'][0]}↔{s['names'][1]} ({s['n_corres']} corres)" for s in weak)
        raise RuntimeError(
            f"graphe de paires déconnecté ({e}) — recouvrement insuffisant. "
            f"Paires les plus faibles : {detail}. Reprendre les photos de ces coins."
        ) from e
    print("[walls] alignement global ok", flush=True)

    pair_scores = _pair_scores(CACHE, image_list, pairs)
    weak_pairs = [s for s in pair_scores if s.get("weak")]

    cam2w = scene.get_im_poses().detach().cpu().numpy().astype(np.float64)
    pts3d, _, confs = scene.get_dense_pts3d(clean_depth=True)
    pts_list = [p.detach().cpu().numpy().astype(np.float64) for p in pts3d]
    confs_list = [np.asarray(c.detach().cpu() if torch.is_tensor(c) else c, dtype=np.float64)
                  for c in confs]
    print(f"[walls] {sum(len(p) for p in pts_list)} points denses extraits", flush=True)

    plan_raw, report = walls_geometry.analyze_walls(
        cam2w, pts_list, confs_list,
        [Path(p).name for p in image_list],
        debug_path=args.out_debug,
    )
    report["pairs"] = pair_scores
    report["weak_pairs"] = [f"{s['names'][0]}↔{s['names'][1]}" for s in weak_pairs]
    if weak_pairs:
        report["warnings"].append(
            f"{len(weak_pairs)} paire(s) faible(s) (< {WEAK_N_CORRES} corres ou "
            f"conf_mean < {WEAK_CONF_MEAN}) — voir report.pairs"
        )
    print(f"[walls] {len(plan_raw['global_plane_info'])} murs, "
          f"metric_check={report['metric_check']['passed']}, "
          f"{len(weak_pairs)} paires faibles", flush=True)

    with open(args.out_json, "w") as f:
        json.dump(plan_raw, f)
    with open(args.out_report, "w") as f:
        json.dump(report, f, indent=1)
    print(f"[walls] plan_raw -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
