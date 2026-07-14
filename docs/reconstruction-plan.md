# Brief — Reconstruction de plan à partir de photos iPhone

> Handoff pour reprise dans Claude Code. Objectif : produire un plan 2D (polygones de pièces, à l'échelle métrique) à partir de quelques photos/vidéo perspective prises à l'iPhone.

## Contexte & contrainte d'entrée

- **Entrée** : quelques photos perspective classiques (iPhone), ou vidéo échantillonnée en frames. Pas de caméra 360.
- **Actif existant** : pipeline **MASt3R-SfM** déjà fonctionnel (poses caméra + point cloud, échelle métrique).
- **Cible v1** : une pièce, potentiellement grande.
- **Cible v2** : plusieurs pièces → plan d'ensemble. On peut organiser les captures par pièce.
- **Sortie visée** : polygone(s) de pièce en mètres, exportable en GeoJSON, consommable depuis un front JS.

## Décision d'architecture retenue

**Approche : Plane-DUSt3R** (papier « Unposed Sparse Views Room Layout Reconstruction in the Age of Pretrain Model », arXiv 2502.16779). Raison du choix : le repo est **bâti directement sur MASt3R** (backbone identique à notre pipeline existant), il prend des images multi-perspectives **non-posées**, prédit les **plans structurels** (murs/sol/plafond) dans un repère commun, et est robuste sur des données in-the-wild (pas seulement du synthétique). Les arêtes de jonction mur/plafond = intersections de ces plans ; leur projection top-down = le polygone de pièce.

- Repo : `https://github.com/justacar/Plane-DUSt3R`
- Alternative écartée : baseline « NonCuboid + MASt3R » (détection de plan mono-vue par image + unification via poses MASt3R). Fonctionne avec notre SfM existant, mais Plane-DUSt3R fait mieux dans le papier, y compris face à NonCuboid avec pose vérité-terrain. On garde ce baseline comme fallback si l'intégration de Plane-DUSt3R coince.

## Point critique : l'assemblage multi-pièces

L'assemblage pièce-par-pièce n'est facile **que si toutes les pièces vivent dans un seul repère global**. SfM indépendant par pièce = repères arbitraires disjoints = recalage inter-pièces fragile (pas de contrainte géométrique sans imagerie partagée).

**Règle** : une seule reconstruction globale, *segmentée* en pièces — pas des reconstructions par pièce qu'on recolle.

1. Capture continue **avec les transitions de portes** (recouvrement visuel aux seuils) → MASt3R enregistre tout dans un repère global unique.
2. Le découpage par pièce sert uniquement à fiabiliser la détection de plans (sous-ensembles propres), mais tout reste dans le repère global.
3. Assemblage = union des polygones dans le repère partagé + soudure des murs mitoyens (plans quasi-coïncidents → mur unique) + placement des ouvertures là où la trajectoire traverse un seuil.

## Gotchas capture (iPhone / sparse views)

- **Recouvrement** : préférer la **vidéo** échantillonnée (frames tous les ~0,5–1 s ou sur seuil de déplacement) plutôt que des snapshots épars → beaucoup plus de contraintes de matching, tout en restant « sparse ».
- **Murs sans texture** : peu de points SfM sur les surfaces plates. C'est la raison de préférer Plane-DUSt3R (prédit les plans) au fit de plans sur point cloud brut.

## Stack cible

- Modèle GPU exposé en service (setup **RunPod A100** existant), appelé en API.
- Post-traitement plans 3D → polygone → GeoJSON.
- Front : **JS (pas TS)**, React (Vite web / Expo mobile) ; rendu du plan sur canvas (Konva déjà utilisé ailleurs).

## Prochaines étapes pour Claude Code

1. **Inspecter le repo Plane-DUSt3R** : dépendances, poids disponibles (`checkpoint-best-onlyencoder.pth`), format d'entrée exact des images, sortie (pointmap de plans + détections 2D).
2. **Brancher notre MASt3R-SfM** : soit appeler Plane-DUSt3R en bout-en-bout (il embarque MASt3R), soit réutiliser nos poses existantes via le chemin NonCuboid+MASt3R. Trancher après lecture du code.
3. **Écrire le post-traitement** : plans 3D structurels → intersections → polygone de pièce (top-down) → GeoJSON métrique. Gérer la fermeture du polygone et le nettoyage (murs colinéaires, bruit).
4. **Packager en service RunPod** : endpoint qui prend N frames → renvoie GeoJSON.
5. **v2 assemblage** : segmentation par pièce dans le repère global, union + soudure murs mitoyens + détection des ouvertures aux seuils.
6. **Front** : composant React qui charge le GeoJSON et le rend (Konva), avec cotes.

## Références

- Plane-DUSt3R — arXiv 2502.16779 — repo `justacar/Plane-DUSt3R`
- MASt3R (Leroy et al., 2024) — backbone, poses à l'échelle métrique
- Baseline alternatif : NonCuboid (Yang et al., 2022) + MASt3R
- Datasets d'entraînement/fine-tuning éventuel : Structured3D
- Voie « nuage de points → plan » si besoin plus tard : FRI-Net, PolyRoom (ECCV 2024), partitionnement d'espace (Fang et al.)