# genrecon-runpod

Photos d'une pièce → mesh 3D texturé (`scene.glb`), via [VGGT](https://github.com/facebookresearch/vggt)
(poses caméra) et [GenRecon](https://github.com/kasothaphie/GenRecon) (reconstruction générative)
sur RunPod Serverless (A100 80GB).

## Usage

```sh
npm install
node recon.mjs <dossier-photos>   # 8-12 photos avec recouvrement
# → scene.glb dans le dossier courant (~7 min, ~0,30 $ de GPU)
```

Secrets attendus dans `.env` : `RUNPOD_API_KEY`, `RUNPOD_ENDPOINT_ID`,
`SUPABASE_URL`, `SUPABASE_SECRET_KEY`, `HF_TOKEN`.

## Architecture

1. La CLI uploade les photos dans le bucket Supabase privé `recon-input` et
   déclenche l'endpoint RunPod (`/run`, puis polling `/status`).
2. Le worker ([worker/handler.py](worker/handler.py)) :
   [vggt_colmap.py](worker/vggt_colmap.py) (feedforward, seuil de confiance
   adaptatif) → COLMAP texte (pycolmap 3.10) → `reconstruct_scene.py --mode
   Iphone` → `chunked_to_glb.py` → upload vers `recon-output` (en morceaux de
   45 Mo, limite Supabase).
3. La CLI réassemble `scene.glb` et le valide (`@gltf-transform/core`).

Les poids (~21 Go : 3 ckpts TUM, VGGT-1B, DINOv3, décodeurs TRELLIS.2) vivent
sur un network volume RunPod monté en `/runpod-volume` (`HF_HOME`, `TORCH_HOME`
et `genrecon/*.pt`). L'image Docker (`ghcr.io/favreau84/genrecon-worker`) est
buildée par [GitHub Actions](.github/workflows/build.yml) — jamais en local.

## Pièges rencontrés (voir l'historique des commits)

- `demo_colmap.py` upstream importe `lightglue` (inutile hors bundle-adjustment) → script local sans BA.
- `pycolmap` doit être **3.10.0** (l'API `Image` a changé ensuite).
- Les ckpts TUM sont des `.pt` nus ; GenRecon attend `<run>/ckpts/*.pt` + `<run>/config.json`
  (configs d'entraînement du repo) → layout reconstruit par symlinks au démarrage du job.
- Seuil de confiance VGGT par défaut (5.0) → 0 point sur scène réelle → seuil adaptatif (quartile supérieur).
- Objets Supabase limités à 50 Mo (plan gratuit) → GLB découpé/réassemblé.

## Coûts

Endpoint serverless : min 0 worker (aucun coût au repos), ~0,30 $/scène.
Volume 50 Go : ~3,50 $/mois tant qu'il existe (poids retéléchargeables en ~30 min).
