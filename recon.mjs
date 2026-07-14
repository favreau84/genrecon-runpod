#!/usr/bin/env node
// CLI : node recon.mjs <dossier-photos>
// Upload des photos vers Supabase Storage, job RunPod Serverless (VGGT + GenRecon),
// rapatriement et validation de scene.glb.

import { createClient } from '@supabase/supabase-js';
import { NodeIO } from '@gltf-transform/core';
import { writeFile } from 'node:fs/promises';
import path from 'node:path';
import process from 'node:process';
import {
  INPUT_BUCKET, OUTPUT_BUCKET,
  loadEnv, need, makeJobId, resolveInputDir, extractFrames, listImages,
  uploadImages, runAndPoll, downloadOutput,
} from './lib/common.mjs';

async function main() {
  const dir = process.argv[2];
  if (!dir) {
    console.error('Usage: node recon.mjs <dossier-test>');
    console.error('  <dossier-test>/input/  → photos (.jpg/.jpeg/.png/.webp)');
    console.error('  <dossier-test>/scene.glb sera écrit à la racine du dossier.');
    console.error('  (compat : si pas de sous-dossier input/, <dossier-test> = dossier de photos,');
    console.error('   scene.glb écrit dans le dossier courant)');
    process.exit(1);
  }
  await loadEnv();
  const { photoDir, outDir } = await resolveInputDir(dir);
  const outPath = path.join(outDir, 'scene.glb');
  const supabase = createClient(need('SUPABASE_URL'), need('SUPABASE_SECRET_KEY'));
  const endpointId = need('RUNPOD_ENDPOINT_ID');
  const runpodKey = need('RUNPOD_API_KEY');

  // Vidéo(s) dans input/ : extraction de frames (régime nominal de GenRecon
  // mode Iphone — beaucoup de vues très recouvrantes). Cap à MAX_FRAMES pour
  // rester dans la mémoire de VGGT.
  const MAX_FRAMES = parseInt(process.env.MAX_FRAMES || '64', 10);
  const FRAME_WIDTH = parseInt(process.env.FRAME_WIDTH || '1920', 10);
  const framesDir = await extractFrames(photoDir, { maxFrames: MAX_FRAMES, frameWidth: FRAME_WIDTH });
  const listDir = framesDir ?? photoDir;
  const files = await listImages(listDir);
  if (files.length < 2) {
    console.error(`Pas assez d'images dans ${listDir} (trouvé : ${files.length})`);
    process.exit(1);
  }
  if (files.length > MAX_FRAMES) {
    console.error(`${files.length} images > ${MAX_FRAMES} (limite mémoire VGGT) — réduisez le nombre de photos`);
    process.exit(1);
  }

  const jobId = makeJobId();
  const prefix = `jobs/${jobId}`;
  console.log(`Job ${jobId} — upload de ${files.length} photos vers ${INPUT_BUCKET}/${prefix} …`);
  await uploadImages(supabase, INPUT_BUCKET, prefix, listDir, files);

  const outputKey = `${prefix}/scene.glb`;
  const output = await runAndPoll({
    endpointId,
    apiKey: runpodKey,
    input: {
      prefix,
      input_bucket: INPUT_BUCKET,
      output_bucket: OUTPUT_BUCKET,
      output_key: outputKey,
    },
    etaHint: 'cold start + pipeline : ~10-20 min',
  });
  console.log('Worker :', JSON.stringify(output));

  console.log(`Téléchargement de ${OUTPUT_BUCKET}/${outputKey} …`);
  const glbBuf = await downloadOutput(supabase, OUTPUT_BUCKET, outputKey, output?.parts ?? 0);
  await writeFile(outPath, glbBuf);

  // Validation : le GLB s'ouvre et contient au moins un mesh non vide
  const doc = await new NodeIO().readBinary(new Uint8Array(glbBuf));
  const meshes = doc.getRoot().listMeshes();
  const totalVertices = meshes
    .flatMap((m) => m.listPrimitives())
    .reduce((n, p) => n + (p.getAttribute('POSITION')?.getCount() ?? 0), 0);
  if (totalVertices === 0) throw new Error('scene.glb ne contient aucun vertex');

  console.log(
    `✅ ${outPath} (${(glbBuf.length / 1e6).toFixed(1)} Mo) — ${meshes.length} mesh(es), ${totalVertices} vertices`
  );
}

main().catch((e) => {
  console.error('\n❌', e.message ?? e);
  process.exit(1);
});
