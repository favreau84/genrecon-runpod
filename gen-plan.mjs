#!/usr/bin/env node
// CLI : node gen-plan.mjs <dossier-ou-video>
// Vidéo (ou photos) d'UNE pièce → frames → Supabase → job RunPod Plane-DUSt3R
// → plan.geojson (polygone métrique) + plan.html (viewer canvas autonome).

import { createClient } from '@supabase/supabase-js';
import { readFile, writeFile, stat } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import { pathToFileURL } from 'node:url';
import {
  INPUT_BUCKET, OUTPUT_BUCKET, VIDEO_EXTS,
  loadEnv, need, makeJobId, resolveInputDir, extractFrames, listImages,
  uploadImages, runAndPoll, downloadOutput,
} from './lib/common.mjs';

// Plane-DUSt3R apparie toutes les images entre elles (O(N²)) : on reste sparse.
const MAX_FRAMES = parseInt(process.env.PLAN_MAX_FRAMES || '12', 10);
const FRAME_WIDTH = parseInt(process.env.PLAN_FRAME_WIDTH || '1280', 10);
const HARD_CAP = 15;

async function main() {
  const arg = process.argv[2];
  if (!arg) {
    console.error('Usage: node gen-plan.mjs <dossier-ou-video>');
    console.error('  <dossier>/input/  → vidéo (.mp4/.mov/.m4v) ou photos d\'UNE pièce');
    console.error('  (ou directement un fichier vidéo, ou un dossier plat de photos)');
    console.error('  Sorties : plan.geojson + plan.html à la racine du dossier.');
    process.exit(1);
  }
  await loadEnv();

  // Entrée : fichier vidéo direct, ou dossier (convention <dir>/input/)
  let photoDir, outDir;
  const st = await stat(arg).catch(() => null);
  if (!st) {
    console.error(`Introuvable : ${arg}`);
    process.exit(1);
  }
  if (st.isFile()) {
    if (!VIDEO_EXTS.has(path.extname(arg).toLowerCase())) {
      console.error(`Fichier non vidéo : ${arg} (attendu ${[...VIDEO_EXTS].join('/')})`);
      process.exit(1);
    }
    photoDir = path.dirname(path.resolve(arg));
    outDir = photoDir;
  } else {
    ({ photoDir, outDir } = await resolveInputDir(arg));
  }

  const supabase = createClient(need('SUPABASE_URL'), need('SUPABASE_SECRET_KEY'));
  const endpointId = need('RUNPOD_PLAN_ENDPOINT_ID');
  const runpodKey = need('RUNPOD_API_KEY');

  const framesDir = await extractFrames(photoDir, { maxFrames: MAX_FRAMES, frameWidth: FRAME_WIDTH });
  const listDir = framesDir ?? photoDir;
  let files = await listImages(listDir);
  if (files.length < 2) {
    console.error(`Pas assez d'images dans ${listDir} (trouvé : ${files.length})`);
    process.exit(1);
  }
  if (files.length > HARD_CAP) {
    // frames vidéo : sous-échantillonnage uniforme ; photos : on refuse
    if (framesDir) {
      const step = files.length / MAX_FRAMES;
      files = Array.from({ length: MAX_FRAMES }, (_, i) => files[Math.floor(i * step)]);
      console.log(`${MAX_FRAMES} frames retenues sur l'extraction (sous-échantillonnage uniforme)`);
    } else {
      console.error(`${files.length} photos > ${HARD_CAP} (Plane-DUSt3R apparie tout : O(N²)) — réduisez la sélection`);
      process.exit(1);
    }
  }

  const jobId = makeJobId();
  const prefix = `jobs/${jobId}`;
  console.log(`Job ${jobId} — upload de ${files.length} images vers ${INPUT_BUCKET}/${prefix} …`);
  await uploadImages(supabase, INPUT_BUCKET, prefix, listDir, files);

  const outputKey = `${prefix}/plan.geojson`;
  const output = await runAndPoll({
    endpointId,
    apiKey: runpodKey,
    input: {
      prefix,
      input_bucket: INPUT_BUCKET,
      output_bucket: OUTPUT_BUCKET,
      output_key: outputKey,
      scale_mode: process.env.PLAN_SCALE_MODE || 'auto',
    },
    etaHint: 'cold start + inférence : ~3-8 min',
  });
  console.log('Worker :', JSON.stringify(output));

  console.log(`Téléchargement de ${OUTPUT_BUCKET}/${outputKey} …`);
  const geoBuf = await downloadOutput(supabase, OUTPUT_BUCKET, outputKey);
  const geojson = JSON.parse(geoBuf.toString('utf8'));

  // plan_raw.json (debug) : best-effort, permet de rejouer layout_to_geojson en local
  try {
    const rawBuf = await downloadOutput(supabase, OUTPUT_BUCKET, `${prefix}/plan_raw.json`);
    await writeFile(path.join(outDir, 'plan_raw.json'), rawBuf);
  } catch {
    console.log('(plan_raw.json indisponible — debug local impossible)');
  }

  // Validation locale minimale
  const room = geojson.features?.find((f) => f.properties?.kind === 'room');
  const ring = room?.geometry?.coordinates?.[0];
  if (!room || !ring || ring.length < 4) throw new Error('GeoJSON sans polygone de pièce valide');
  if (!(room.properties.area_m2 > 0.5)) throw new Error(`aire invalide : ${room.properties.area_m2} m²`);

  const geoPath = path.join(outDir, 'plan.geojson');
  await writeFile(geoPath, JSON.stringify(geojson, null, 2));

  const htmlPath = path.join(outDir, 'plan.html');
  await writeFile(htmlPath, await renderViewer(geojson));

  const p = room.properties;
  console.log(
    `✅ ${geoPath} — ${p.n_walls} murs, ${p.area_m2} m², h. plafond ${p.ceiling_height_m ?? '?'} m ` +
    `(échelle : ${p.scale_mode}${p.closed ? '' : ', pièce partiellement filmée'})`
  );
  if (p.warnings?.length) console.log(`⚠ ${p.warnings.join(' | ')}`);
  console.log(`→ ouvrir ${htmlPath}`);
}

export async function renderViewer(geojson) {
  const template = await readFile(new URL('./lib/plan-viewer.template.html', import.meta.url), 'utf8');
  const marker = '/*__GEOJSON__*/null';
  if (!template.includes(marker)) throw new Error('marqueur __GEOJSON__ absent du template viewer');
  return template.replace(marker, JSON.stringify(geojson));
}

if (process.argv[1] && import.meta.url === pathToFileURL(process.argv[1]).href) {
  main().catch((e) => {
    console.error('\n❌', e.message ?? e);
    process.exit(1);
  });
}
