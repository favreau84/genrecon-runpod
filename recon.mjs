#!/usr/bin/env node
// CLI : node recon.mjs <dossier-photos>
// Upload des photos vers Supabase Storage, job RunPod Serverless (VGGT + GenRecon),
// rapatriement et validation de scene.glb.

import { createClient } from '@supabase/supabase-js';
import { NodeIO } from '@gltf-transform/core';
import { readFile, readdir, writeFile } from 'node:fs/promises';
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
    console.error('Usage: node recon.mjs <dossier-photos>');
    process.exit(1);
  }
  await loadEnv();
  const supabase = createClient(need('SUPABASE_URL'), need('SUPABASE_SECRET_KEY'));
  const endpointId = need('RUNPOD_ENDPOINT_ID');
  const runpodKey = need('RUNPOD_API_KEY');

  const files = (await readdir(dir)).filter((f) => IMAGE_EXTS.has(path.extname(f).toLowerCase()));
  if (files.length < 2) {
    console.error(`Pas assez d'images dans ${dir} (trouvé : ${files.length})`);
    process.exit(1);
  }

  const jobId = `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
  const prefix = `jobs/${jobId}`;
  console.log(`Job ${jobId} — upload de ${files.length} photos vers ${INPUT_BUCKET}/${prefix} …`);

  for (const f of files.sort()) {
    const buf = await readFile(path.join(dir, f));
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
  const { data, error } = await supabase.storage.from(OUTPUT_BUCKET).download(outputKey);
  if (error) throw new Error(`Download scene.glb : ${error.message}`);
  const glbBuf = Buffer.from(await data.arrayBuffer());
  const outPath = path.resolve('scene.glb');
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
