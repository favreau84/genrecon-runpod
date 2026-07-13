# Mission

Implémenter le pipeline "photos → mesh GLB" décrit dans docs/plan-technique.md :
une CLI locale `node recon.mjs <dossier-photos>` qui uploade les images vers
Supabase Storage, déclenche un endpoint RunPod Serverless exécutant
VGGT (poses COLMAP) puis GenRecon (github.com/kasothaphie/GenRecon), et
rapatrie scene.glb. Suivre le plan technique ; en cas de conflit entre ce
fichier et le plan, ce fichier prime.

# Environnement et accès

- Secrets dans .env (ne jamais les committer, ne jamais les afficher en clair).
- Supabase : SUPABASE_URL + SUPABASE_SECRET_KEY (clé nouvelle génération
  sb_secret_..., pas un JWT). OBLIGATOIRE : passer par les SDK officiels
  (supabase-js pour la CLI, supabase-py pour le worker) — jamais d'appels REST
  bruts, car ces clés ne sont pas valides en header Authorization: Bearer.
- RunPod : tout se fait par API REST (https://rest.runpod.io/v1) avec
  RUNPOD_API_KEY — création de network volume, d'endpoint, jobs, logs.
- Build Docker : JAMAIS en local (Mac ARM). Créer .github/workflows/build.yml
  (docker/build-push-action, platform linux/amd64, push vers
  ghcr.io/<owner>/genrecon-worker). Déclencher et surveiller avec `gh run`.
- Le build compile des extensions CUDA : prévoir 60-90 min de CI. Ne pas
  relancer un build sans avoir modifié le Dockerfile.

# Règles budget (STRICTES — crédit total : 10 $)

1. Network volume : 50 Go max, un seul.
2. Pour télécharger les poids sur le volume : pod le MOINS CHER disponible
   avec le volume attaché ; le TERMINER immédiatement après (vérifier via API
   qu'il est bien terminé). Poids à télécharger : les 3 checkpoints
   kaldir.vc.cit.tum.de/genrecon/{sparse_structure,shape_slat,texture_slat}.pt
   + microsoft/TRELLIS.2-4B et facebook/VGGT-1B via huggingface-cli (HF_TOKEN).
3. Endpoint serverless : GPU A100 80GB, min workers 0, max workers 1,
   idle timeout 5 s, execution timeout 1800 s, FlashBoot on.
4. Maximum 4 runs de test GPU. Si le 4e échoue : STOP, rédiger un rapport
   d'état (ce qui marche, ce qui bloque, hypothèses) et me rendre la main.
5. Vérifier le crédit restant (API RunPod) avant chaque run ; si < 3 $, STOP.

# Définition de "terminé"

`node recon.mjs ./fixtures/test-scene/` (8-10 photos d'une pièce que je
fournirai dans fixtures/) produit un fichier scene.glb valide (s'ouvre sans
erreur, contient un mesh non vide — vérifiable avec le package Node
@gltf-transform/core en inspection locale).

# Style

- JS (pas TS) pour la CLI, Node 20+, modules ES.
- Python uniquement côté worker RunPod.
- Commits atomiques, messages clairs. Un README court à la fin.

# Points de vigilance connus (issus de l'étude préalable)

- Le repo GenRecon est très récent (code publié le 29.06.2026) : lire son
  README/code AVANT d'écrire le Dockerfile ; le setup officiel passe par
  setup.sh (flags --flash-attn --nvdiffrast --nvdiffrec --cumesh --o-voxel
  --flexgemm) ; fixer TORCH_CUDA_ARCH_LIST="8.0" (A100).
- Entrée GenRecon : format COLMAP (mode Scannet_colmap). Vérifier
  l'arborescence exacte attendue (sparse/0, .bin vs .txt) dans le code source.
- VGGT fournit un export COLMAP (demo_colmap.py) — l'utiliser pour les poses.
- Si OOM : réduire la résolution/taille des chunks avant de changer de GPU.