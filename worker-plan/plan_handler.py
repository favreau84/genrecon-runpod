"""Handler RunPod Serverless : images d'une pièce (Supabase) -> Plane-DUSt3R
-> plan.geojson + plan_raw.json (Supabase).

Entrée job :
    {
        "input": {
            "prefix": "jobs/<id>",                 # préfixe des images dans recon-input
            "input_bucket": "recon-input",          # optionnel
            "output_bucket": "recon-output",        # optionnel
            "output_key": "jobs/<id>/plan.geojson", # optionnel
            "scale_mode": "auto"                    # auto | metric | none
        }
    }

Poids attendus sur le network volume (monté en /runpod-volume) :
    /runpod-volume/planedust3r/checkpoint-best-onlyencoder.pth
    /runpod-volume/planedust3r/Structured3D_pretrained.pt
    /runpod-volume/hf -> HF_HOME (MASt3R métrique, pour le fallback échelle)
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

import runpod  # noqa: E402
from supabase import create_client  # noqa: E402

PLANEDUST3R = Path(os.environ.get("PLANEDUST3R_DIR", "/opt/planedust3r"))
WORKER = Path(__file__).parent

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp"}
MAX_IMAGES = 16  # scene_graph='complete' : O(N²) paires à 512px


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


def _upload_json(sb, bucket, key, path: Path, content_type):
    sb.storage.from_(bucket).upload(
        key, path.read_bytes(), {"content-type": content_type, "upsert": "true"}
    )
    print(f"[output] {path.name} -> {bucket}/{key}", flush=True)


def handler(job):
    inp = job.get("input") or {}
    prefix = inp["prefix"].strip("/")
    input_bucket = inp.get("input_bucket", "recon-input")
    output_bucket = inp.get("output_bucket", "recon-output")
    output_key = inp.get("output_key", f"{prefix}/plan.geojson")
    scale_mode = str(inp.get("scale_mode", "auto")).lower()
    job_id = prefix.rsplit("/", 1)[-1]

    for ckpt in ("checkpoint-best-onlyencoder.pth", "Structured3D_pretrained.pt"):
        if not (VOLUME / "planedust3r" / ckpt).exists():
            return {"error": f"Checkpoint manquant sur le volume : planedust3r/{ckpt}"}

    scene = Path("/tmp/plan_scene")
    shutil.rmtree(scene, ignore_errors=True)
    plan_raw = scene / "plan_raw.json"
    plan_geojson = scene / "plan.geojson"
    timings = {}

    try:
        n_images = _download_images(_sb(), input_bucket, prefix, scene / "images")
        if n_images > MAX_IMAGES:
            raise RuntimeError(
                f"{n_images} images > {MAX_IMAGES} (Plane-DUSt3R apparie tout : O(N²)) — "
                "réduire l'échantillonnage côté CLI"
            )

        # 1. Inférence Plane-DUSt3R (headless) -> plan_raw.json
        timings["infer_s"] = _run(
            [
                sys.executable, WORKER / "planedust3r_infer.py",
                "--image_dir", scene / "images",
                "--out_json", plan_raw,
                "--scale_mode", scale_mode,
            ],
            cwd=PLANEDUST3R,
            log_name="planedust3r",
        )

        # 2. Post-traitement géométrique -> plan.geojson
        timings["layout_s"] = _run(
            [
                sys.executable, WORKER / "layout_to_geojson.py",
                "--in_json", plan_raw,
                "--out", plan_geojson,
                "--job_id", job_id,
            ],
            cwd=WORKER,
            log_name="layout",
        )

        # 3. Upload : le GeoJSON + le raw (debug local sans re-run GPU)
        sb = _sb()
        raw_key = f"{prefix}/plan_raw.json"
        _upload_json(sb, output_bucket, raw_key, plan_raw, "application/json")
        _upload_json(sb, output_bucket, output_key, plan_geojson, "application/geo+json")

        import json
        props = json.loads(plan_geojson.read_text())["features"][0]["properties"]
        return {
            "output_bucket": output_bucket,
            "output_key": output_key,
            "raw_key": raw_key,
            "area_m2": props["area_m2"],
            "perimeter_m": props["perimeter_m"],
            "ceiling_height_m": props["ceiling_height_m"],
            "n_walls": props["n_walls"],
            "n_doors": props["n_doors"],
            "n_windows": props["n_windows"],
            "closed": props["closed"],
            "scale_mode": props["scale_mode"],
            "scale_factor": props["scale_factor"],
            "warnings": props["warnings"],
            "n_images": n_images,
            "timings": {k: round(v) for k, v in timings.items()},
        }
    except Exception as e:
        traceback.print_exc()
        # même en échec, tenter de remonter plan_raw.json pour le debug local
        try:
            if plan_raw.exists():
                _upload_json(_sb(), output_bucket, f"{prefix}/plan_raw.json", plan_raw, "application/json")
        except Exception:
            pass
        return {"error": str(e), "timings": {k: round(v) for k, v in timings.items()}}
    finally:
        shutil.rmtree(scene, ignore_errors=True)


runpod.serverless.start({"handler": handler})
