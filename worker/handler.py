"""Handler RunPod Serverless : photos (Supabase) -> VGGT -> GenRecon -> scene.glb (Supabase).

Entrée job :
    {
        "input": {
            "prefix": "jobs/<id>",              # préfixe des images dans recon-input
            "input_bucket": "recon-input",       # optionnel
            "output_bucket": "recon-output",     # optionnel
            "output_key": "jobs/<id>/scene.glb", # optionnel
            "num_imgs_per_scene": 999,           # optionnel
            "proj_batch_voxels": 2048            # optionnel (baisser si OOM)
        }
    }

Poids attendus sur le network volume (montés en /runpod-volume) :
    /runpod-volume/genrecon/{sparse_structure,shape_slat,texture_slat}.pt
    /runpod-volume/hf      -> HF_HOME (TRELLIS.2-4B, TRELLIS-image-large, DINOv3)
    /runpod-volume/torch   -> TORCH_HOME (VGGT-1B model.pt, cache torch.hub)
"""

import collections
import os
import shutil
import subprocess
import sys
import time
import traceback
from pathlib import Path

VOLUME = Path(os.environ.get("RUNPOD_VOLUME_PATH", "/runpod-volume"))
os.environ.setdefault("HF_HOME", str(VOLUME / "hf"))
os.environ.setdefault("TORCH_HOME", str(VOLUME / "torch"))
# Pas de HF_HUB_OFFLINE : si un fichier manque dans le cache du volume, il est
# retéléchargé (HF_TOKEN fourni en env var d'endpoint) et persiste sur le volume.

import runpod  # noqa: E402
from supabase import create_client  # noqa: E402

GENRECON = Path("/opt/GenRecon")
VGGT_DIR = Path("/opt/vggt")
CKPT_DIR = VOLUME / "genrecon"

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}


# load_train_config() attend une arborescence de run d'entraînement
# (<run>/ckpts/model.pt + <run>/config.json) ; le serveur TUM ne fournit que
# les .pt. On reconstruit le layout avec des symlinks + les configs du repo.
_STAGES = {
    "ss": ("sparse_structure.pt", "configs/gen/ss_flow_img/genrecon.json"),
    "shape": ("shape_slat.pt", "configs/gen/slat_flow_img2shape/genrecon_512.json"),
    "tex": ("texture_slat.pt", "configs/gen/slat_flow_imgshape2tex/genrecon_512.json"),
}


def _ckpt_layout():
    root = Path("/tmp/ckpt_layout")
    shutil.rmtree(root, ignore_errors=True)
    paths = {}
    for stage, (ckpt_name, config_rel) in _STAGES.items():
        src = CKPT_DIR / ckpt_name
        if not src.exists():
            raise RuntimeError(f"Checkpoint manquant sur le volume : {src}")
        run_dir = root / stage
        (run_dir / "ckpts").mkdir(parents=True)
        (run_dir / "ckpts" / ckpt_name).symlink_to(src)
        shutil.copy(GENRECON / config_rel, run_dir / "config.json")
        paths[stage] = run_dir / "ckpts" / ckpt_name
    return paths


def _sb():
    return create_client(os.environ["SUPABASE_URL"], os.environ["SUPABASE_SECRET_KEY"])


def _run(cmd, cwd, log_name):
    """Lance un sous-processus : stream vers les logs RunPod + garde la fin de la
    sortie pour la remonter dans l'erreur du job (les logs console ne sont pas
    accessibles par API)."""
    print(f"[{log_name}] $ {' '.join(str(c) for c in cmd)}", flush=True)
    t0 = time.time()
    tail = collections.deque(maxlen=60)
    proc = subprocess.Popen(
        [str(c) for c in cmd],
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=os.environ.copy(),
    )
    for line in proc.stdout:
        print(f"[{log_name}] {line}", end="", flush=True)
        tail.append(line)
    proc.wait()
    dt = time.time() - t0
    print(f"[{log_name}] exit={proc.returncode} en {dt:.0f}s", flush=True)
    if proc.returncode != 0:
        raise RuntimeError(
            f"{log_name} a échoué (exit {proc.returncode}). Fin de sortie :\n" + "".join(tail)[-3500:]
        )
    return dt


def _download_images(sb, bucket, prefix, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    entries = sb.storage.from_(bucket).list(prefix, {"limit": 1000})
    names = [
        e["name"]
        for e in entries
        if e.get("id") is not None and Path(e["name"]).suffix.lower() in IMAGE_EXTS
    ]
    if not names:
        raise RuntimeError(f"Aucune image sous {bucket}/{prefix}")
    for name in sorted(names):
        data = sb.storage.from_(bucket).download(f"{prefix}/{name}")
        (dest / name).write_bytes(data)
    print(f"[input] {len(names)} images téléchargées depuis {bucket}/{prefix}", flush=True)
    return len(names)


def _sparse_bin_to_txt(sparse_dir: Path, txt_dir: Path):
    """VGGT écrit du COLMAP binaire dans sparse/ ; GenRecon (mode Iphone) lit du texte."""
    import pycolmap

    rec = pycolmap.Reconstruction(str(sparse_dir))
    txt_dir.mkdir(parents=True, exist_ok=True)
    rec.write_text(str(txt_dir))
    n_pts = rec.num_points3D() if callable(getattr(rec, "num_points3D", None)) else len(rec.points3D)
    print(f"[colmap] {len(rec.images)} images, {n_pts} points3D -> {txt_dir}", flush=True)
    if n_pts < 100:
        raise RuntimeError(
            f"VGGT n'a produit que {n_pts} points3D — nuage trop pauvre pour le "
            "chunker GenRecon (photos trop peu texturées ou chevauchement insuffisant ?)"
        )
    return n_pts


def handler(job):
    inp = job.get("input") or {}
    prefix = inp["prefix"].strip("/")
    input_bucket = inp.get("input_bucket", "recon-input")
    output_bucket = inp.get("output_bucket", "recon-output")
    output_key = inp.get("output_key", f"{prefix}/scene.glb")
    num_imgs = int(inp.get("num_imgs_per_scene", 999))
    proj_batch_voxels = int(inp.get("proj_batch_voxels", 2048))

    ckpts = _ckpt_layout()

    scene = Path("/tmp/scene")
    shutil.rmtree(scene, ignore_errors=True)
    out_dir = scene / "out"
    timings = {}

    try:
        # 1. Photos depuis Supabase — VGGT lit <scene>/images/, GenRecon <scene>/rgb/
        n_images = _download_images(_sb(), input_bucket, prefix, scene / "images")
        (scene / "rgb").mkdir()
        for f in (scene / "images").iterdir():
            os.link(f, scene / "rgb" / f.name)

        # 2. VGGT feedforward -> poses + nuage de points, COLMAP binaire dans sparse/
        # (script local : demo_colmap.py upstream importe lightglue via track_predict)
        timings["vggt_s"] = _run(
            [sys.executable, "/opt/worker/vggt_colmap.py", "--scene_dir", scene],
            cwd=VGGT_DIR,
            log_name="vggt",
        )

        # 3. Conversion binaire -> texte, arborescence attendue par --mode Iphone
        n_points3d = _sparse_bin_to_txt(scene / "sparse", scene / "colmap")

        # 4. GenRecon (flags iPhone du README ; cwd=GenRecon pour les configs relatives)
        timings["genrecon_s"] = _run(
            [
                sys.executable, GENRECON / "reconstruct_scene.py",
                "--mode", "Iphone",
                "--path", scene,
                "--output_path", out_dir,
                "--colmap_subdir", "colmap",
                "--ss_ckpt", ckpts["ss"],
                "--shape_ckpt", ckpts["shape"],
                "--tex_ckpt", ckpts["tex"],
                "--num_imgs_per_scene", num_imgs,
                "--chunk_size_factor", "1.08",
                "--stat_std_ratio", "3.0",
                "--radius_nb_points", "7",
                "--radius_m", "0.2",
                "--pipeline_config", "configs/pipelines/texture.json",
                "--proj_batch_voxels", proj_batch_voxels,
            ],
            cwd=GENRECON,
            log_name="genrecon",
        )

        # 5. Bake GLB
        timings["glb_s"] = _run(
            [
                sys.executable, GENRECON / "chunked_to_glb.py",
                "--inputs", out_dir / "to_glb_inputs.pt",
                "--chunk_inputs", out_dir / "chunk_inputs.pt",
                "--output_dir", out_dir,
            ],
            cwd=GENRECON,
            log_name="glb",
        )

        glb = out_dir / "scene.glb"
        if not glb.exists() or glb.stat().st_size == 0:
            raise RuntimeError("chunked_to_glb.py n'a pas produit de scene.glb non vide")

        # 6. Upload du résultat
        sb = _sb()
        with glb.open("rb") as f:
            sb.storage.from_(output_bucket).upload(
                output_key, f, {"content-type": "model/gltf-binary", "upsert": "true"}
            )
        size_mb = glb.stat().st_size / 1e6
        print(f"[output] scene.glb ({size_mb:.1f} Mo) -> {output_bucket}/{output_key}", flush=True)

        return {
            "output_bucket": output_bucket,
            "output_key": output_key,
            "glb_size_mb": round(size_mb, 2),
            "n_images": n_images,
            "n_points3d": n_points3d,
            "timings": {k: round(v) for k, v in timings.items()},
        }
    except Exception as e:
        traceback.print_exc()
        return {"error": str(e), "timings": {k: round(v) for k, v in timings.items()}}
    finally:
        shutil.rmtree(scene, ignore_errors=True)


runpod.serverless.start({"handler": handler})
