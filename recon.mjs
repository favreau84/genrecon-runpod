#!/usr/bin/env node
// CLI : node recon.mjs <dossier-photos>
// Upload des photos vers Supabase Storage, job RunPod Serverless (VGGT + GenRecon),
// rapatriement et validation de scene.glb.

import { createClient } from '@supabase/supabase-js';
import { NodeIO } from '@gltf-transform/core';
import { readFile, readdir, writeFile, stat, mkdir, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp']);
const INPUT_BUCKET = 'recon-input';
const OUTPUT_BUCKET = 'recon-output';
const POLL_MS = 10_000;

function loadEnv() {
  // .env minimaliste : lignes KEY=VALUE, pas de dépendance externe
  return readFile(new URL('.env', import.meta.url), 'utf8').then((txt) => {
    for (const line of txt.split('\n')) {
      const m = line.match(/^([A-Z_]+)=(.*)$/);
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
    }
  });
}

function need(name) {
  const v = process.env[name];
  if (!v) {
    console.error(`Variable manquante dans .env : ${name}`);
    process.exit(1);
  }
  return v;
}

const contentTypes = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
};

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
  // Layout "un dossier par test" : photos dans <dir>/input, GLB à la racine de <dir>
  let photoDir = dir;
  let outPath = path.resolve('scene.glb');
  const inputSub = path.join(dir, 'input');
  if (await stat(inputSub).then((s) => s.isDirectory()).catch(() => false)) {
    photoDir = inputSub;
    outPath = path.resolve(dir, 'scene.glb');
  }
  const supabase = createClient(need('SUPABASE_URL'), need('SUPABASE_SECRET_KEY'));
  const endpointId = need('RUNPOD_ENDPOINT_ID');
  const runpodKey = need('RUNPOD_API_KEY');

  // Vidéo(s) dans input/ : extraction de frames (régime nominal de GenRecon
  // mode Iphone — beaucoup de vues très recouvrantes). Cap à MAX_FRAMES pour
  // rester dans la mémoire de VGGT.
  const VIDEO_EXTS = new Set(['.mp4', '.mov', '.m4v']);
  const MAX_FRAMES = parseInt(process.env.MAX_FRAMES || '64', 10);
  const FRAME_WIDTH = parseInt(process.env.FRAME_WIDTH || '1920', 10);
  const entries = await readdir(photoDir);
  const videos = entries.filter((f) => VIDEO_EXTS.has(path.extname(f).toLowerCase()));
  let framesDir = null;
  if (videos.length > 0) {
    framesDir = path.join(photoDir, '.frames');
    await rm(framesDir, { recursive: true, force: true });
    await mkdir(framesDir, { recursive: true });
    for (const [vi, v] of videos.entries()) {
      const src = path.join(photoDir, v);
      const dur = parseFloat(
        execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src], {
          encoding: 'utf8',
        })
      );
      const fps = Math.min(3, Math.max(0.5, MAX_FRAMES / videos.length / Math.max(dur, 1)));
      console.log(`Vidéo ${v} (${dur.toFixed(0)}s) → extraction à ${fps.toFixed(2)} img/s …`);
      execFileSync('ffmpeg', [
        '-loglevel', 'error', '-i', src,
        '-vf', `fps=${fps},scale='min(${FRAME_WIDTH},iw)':-2`,
        '-q:v', '2', path.join(framesDir, `v${vi}_%04d.jpg`),
      ]);
    }
  }
  const listDir = framesDir ?? photoDir;
  const files = (await readdir(listDir)).filter((f) => IMAGE_EXTS.has(path.extname(f).toLowerCase()));
  if (files.length < 2) {
    console.error(`Pas assez d'images dans ${listDir} (trouvé : ${files.length})`);
    process.exit(1);
  }
  if (files.length > MAX_FRAMES) {
    console.error(`${files.length} images > ${MAX_FRAMES} (limite mémoire VGGT) — réduisez le nombre de photos`);
    process.exit(1);
  }

  const jobId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const prefix = `jobs/${jobId}`;
  console.log(`Job ${jobId} — upload de ${files.length} photos vers ${INPUT_BUCKET}/${prefix} …`);

  for (const f of files.sort()) {
    const buf = await readFile(path.join(listDir, f));
    const { error } = await supabase.storage
      .from(INPUT_BUCKET)
      .upload(`${prefix}/${f}`, buf, {
        contentType: contentTypes[path.extname(f).toLowerCase()],
        upsert: true,
      });
    if (error) throw new Error(`Upload ${f} : ${error.message}`);
    process.stdout.write('.');
  }
  console.log(' ok');

  const outputKey = `${prefix}/scene.glb`;
  console.log('Déclenchement du job RunPod …');
  const runRes = await fetch(`https://api.runpod.ai/v2/${endpointId}/run`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${runpodKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({
      input: {
        prefix,
        input_bucket: INPUT_BUCKET,
        output_bucket: OUTPUT_BUCKET,
        output_key: outputKey,
      },
    }),
  });
  if (!runRes.ok) throw new Error(`RunPod /run : HTTP ${runRes.status} ${await runRes.text()}`);
  const { id: runId } = await runRes.json();
  console.log(`Run ${runId} — polling toutes les ${POLL_MS / 1000}s (cold start + pipeline : ~10-20 min) …`);

  const t0 = Date.now();
  let output;
  for (;;) {
    await new Promise((r) => setTimeout(r, POLL_MS));
    const res = await fetch(`https://api.runpod.ai/v2/${endpointId}/status/${runId}`, {
      headers: { Authorization: `Bearer ${runpodKey}` },
    });
    if (!res.ok) throw new Error(`RunPod /status : HTTP ${res.status}`);
    const st = await res.json();
    const elapsed = Math.round((Date.now() - t0) / 1000);
    process.stdout.write(`\r[${elapsed}s] statut : ${st.status}        `);
    if (st.status === 'COMPLETED') {
      output = st.output;
      console.log();
      break;
    }
    if (['FAILED', 'CANCELLED', 'TIMED_OUT'].includes(st.status)) {
      console.log();
      throw new Error(`Job ${st.status} : ${JSON.stringify(st, null, 2)}`);
    }
  }

  if (output?.error) throw new Error(`Le worker a renvoyé une erreur : ${output.error}`);
  console.log('Worker :', JSON.stringify(output));

  console.log(`Téléchargement de ${OUTPUT_BUCKET}/${outputKey} …`);
  let glbBuf;
  if (output?.parts > 0) {
    // GLB > limite d'objet Supabase : le worker l'a découpé en .partNNN
    const chunks = [];
    for (let i = 0; i < output.parts; i++) {
      const key = `${outputKey}.part${String(i).padStart(3, '0')}`;
      const { data, error } = await supabase.storage.from(OUTPUT_BUCKET).download(key);
      if (error) throw new Error(`Download ${key} : ${error.message}`);
      chunks.push(Buffer.from(await data.arrayBuffer()));
      process.stdout.write(`\rpart ${i + 1}/${output.parts}   `);
    }
    console.log();
    glbBuf = Buffer.concat(chunks);
  } else {
    const { data, error } = await supabase.storage.from(OUTPUT_BUCKET).download(outputKey);
    if (error) throw new Error(`Download scene.glb : ${error.message}`);
    glbBuf = Buffer.from(await data.arrayBuffer());
  }
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
