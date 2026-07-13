# GenRecon piloté par Claude Code — préliminaires manuels puis agents autonomes

**Principe :** vous faites à la main uniquement ce qu'un agent ne peut pas faire
(créations de comptes, saisie de CB, génération de clés), vous posez des garde-fous
de budget, puis Claude Code fait tout le reste via API/CLI : écrire le code,
builder l'image, créer le volume et l'endpoint RunPod, tester, itérer.

**Budget setup : ≤ 10 € (~11 $).** Répartition prévue :

| Poste | Coût estimé |
|---|---|
| Crédit RunPod initial | 10 $ (dont ~la moitié consommée au setup) |
| Network volume 50 Go | ~3,50 $/mois (0,07 $/Go/mois) |
| Pod temporaire téléchargement poids (~1 h, GPU cheap) | ~0,30 $ |
| 3-4 runs de test A100 (~10-15 min chacun) | ~1,50 $ |
| Build Docker (GitHub Actions, repo public) | 0 $ |
| **Total setup consommé** | **~5-6 $** ✅ |

---

## Partie A — Étapes manuelles (45 min, une seule fois)

### A1. RunPod (le seul compte payant)

- [ ] Créer un compte sur runpod.io
- [ ] **Billing → ajouter 10 $ de crédit** (paiement one-shot, PAS d'auto-refill —
      c'est votre garde-fou budget : quand c'est vide, tout s'arrête)
- [ ] Settings → API Keys → créer une clé **All permissions** (l'agent devra créer
      volume + endpoint + lancer des jobs) → noter `RUNPOD_API_KEY`

### A2. GitHub (build Docker gratuit)

Le build de l'image (compilation CUDA, 45-90 min) se fait sur GitHub Actions :
gratuit en repo **public**, et pilotable par l'agent via `gh` CLI.

- [ ] Compte GitHub (existant) + `gh auth login` fait sur le Mac — c'est tout
      pour cette étape : le repo lui-même sera créé en Partie C par
      `gh repo create` depuis le dossier local.
- [ ] À savoir : l'image ira sur **GHCR** (registry GitHub, gratuit, aucune
      inscription supplémentaire). Une fois le repo créé (Partie C), vérifier
      Settings du repo → Actions → General → Workflow permissions →
      **Read and write** (pour que le workflow puisse push sur ghcr.io).

### A3. Supabase (vous l'avez déjà)

Votre projet utilise le nouveau système de clés (`sb_secret_...` remplace la
`service_role` legacy — drop-in, les SDK l'acceptent sans modification).

- [ ] Choisir/réutiliser un projet → noter le **projectId** ; l'URL s'en déduit :
      `SUPABASE_URL=https://<projectId>.supabase.co`
- [ ] Settings → API Keys → révéler/créer une **Secret key** → noter
      `SUPABASE_SECRET_KEY=sb_secret_...`
- [ ] Créer les 2 buckets privés : `recon-input`, `recon-output`
      (2 min à la main, ou laissez l'agent le faire via l'API management)

⚠️ Particularité des clés `sb_secret_` : elles ne passent pas dans le header
`Authorization: Bearer` (ce ne sont plus des JWT) mais dans le header `apikey`.
Les SDK officiels (supabase-js côté CLI, supabase-py côté worker) gèrent ça
automatiquement — règle pour l'agent : toujours passer par les SDK, jamais
d'appels REST bruts vers Supabase.

### A4. Hugging Face (gratuit)

- [ ] Compte HF + token **read** (Settings → Access Tokens) → noter `HF_TOKEN`
- [ ] Visiter la page `microsoft/TRELLIS.2-4B` et accepter les conditions si le
      modèle est "gated" (clic manuel obligatoire, un agent ne peut pas le faire)

### A5. Poser les secrets

Sur le Mac, dans le futur dossier de travail :

```bash
mkdir -p ~/dev/genrecon-runpod && cd ~/dev/genrecon-runpod
cat > .env <<'EOF'
RUNPOD_API_KEY=...
SUPABASE_URL=https://<projectId>.supabase.co
SUPABASE_SECRET_KEY=sb_secret_...
HF_TOKEN=...
EOF
echo ".env" >> .gitignore
```

(Le secret GitHub `HF_TOKEN` pour le workflow de build sera posé en Partie C,
une fois le repo créé — `gh secret set` agit sur un repo distant, pas sur un
dossier local.)

### A6. Garde-fous anti-dérapage budget

À imposer et à écrire noir sur blanc dans le CLAUDE.md (Partie B) :

1. Crédit RunPod prépayé 10 $, **auto-refill OFF** → plafond physique.
2. Endpoint : **max workers = 1**, **idle timeout = 5 s**, **execution timeout = 1800 s**.
3. Volume : **50 Go max**.
4. Pods temporaires : GPU le moins cher dispo, et l'agent doit **terminer le pod**
   dès la fin du téléchargement des poids (règle explicite).
5. L'agent s'arrête et rend la main après **4 runs de test** infructueux.

---

## Partie B — Préparer le handoff (15 min)

### B1. Structure initiale du repo

```
genrecon-runpod/
├── CLAUDE.md            # la mission + les règles (voir B2)
├── .env                 # secrets locaux (gitignoré)
├── docs/
│   └── plan-technique.md   # copie du plan technique détaillé (l'autre fichier)
└── .github/workflows/   # l'agent créera build.yml
```

Copiez `plan-genrecon-runpod.md` dans `docs/plan-technique.md` : c'est la
spécification technique que l'agent suivra (Dockerfile, handler.py, pipeline.py,
CLI Node, checkpoints TUM, reconstruct_scene.py --mode Scannet_colmap, etc.).

### B2. CLAUDE.md (à coller tel quel, puis ajuster)

```markdown
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
```

### B3. Fixtures de test

- [ ] Prendre 8-10 photos d'une pièce chez vous (recouvrement entre vues,
      couvrir les coins) → `fixtures/test-scene/`. À faire vous-même : l'agent
      ne peut pas photographier votre salon.

---

## Partie C — Lancement (5 min)

```bash
cd ~/dev/genrecon-runpod
git init && git add -A && git commit -m "bootstrap"
gh repo create genrecon-runpod --public --source=. --push

# Maintenant que le repo existe : poser le secret pour le workflow de build
gh secret set HF_TOKEN --body "hf_..."   # seul secret nécessaire au build ;
                                         # les autres iront en env vars RunPod

claude
```

Prompt de départ :

> Lis CLAUDE.md et docs/plan-technique.md, puis implémente la mission de bout
> en bout. Commence par cloner et lire le code de GenRecon pour valider les
> hypothèses du plan (format COLMAP, sorties, setup.sh), et présente-moi ton
> plan d'exécution en 10 lignes avant de commencer à dépenser du crédit RunPod.

Le "présente-moi ton plan avant de dépenser" vous donne un point de contrôle
humain juste avant les euros. Ensuite, laissez tourner ; les règles du CLAUDE.md
bornent la dépense (plafond physique = crédit prépayé sans auto-refill).

**Sessions longues :** le build CI (60-90 min) et les runs GPU (10-20 min) sont
des attentes passives — pensez à lancer Claude Code dans tmux, ou utilisez la
poursuite de session (`claude --continue`) si la session expire.

---

## Ce qui restera à votre charge après le setup

- Recharger le crédit RunPod quand nécessaire (~0,25-0,50 $/scène).
- Le volume coûte ~3,50 $/mois tant qu'il existe — le supprimer si vous
  mettez le projet en pause (les poids se retéléchargent en ~30 min).
