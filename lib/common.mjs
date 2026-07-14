// Utilitaires partagés entre les CLI (recon.mjs, gen-plan.mjs) :
// .env, extraction de frames vidéo, upload Supabase, job RunPod, download.

import { readFile, readdir, stat, mkdir, rm } from 'node:fs/promises';
import { execFileSync } from 'node:child_process';
import path from 'node:path';
import process from 'node:process';

export const IMAGE_EXTS = new Set(['.jpg', '.jpeg', '.png', '.webp']);
export const VIDEO_EXTS = new Set(['.mp4', '.mov', '.m4v']);
export const INPUT_BUCKET = 'recon-input';
export const OUTPUT_BUCKET = 'recon-output';

export const CONTENT_TYPES = {
  '.jpg': 'image/jpeg',
  '.jpeg': 'image/jpeg',
  '.png': 'image/png',
  '.webp': 'image/webp',
};

export function loadEnv() {
  // .env minimaliste : lignes KEY=VALUE, pas de dépendance externe
  return readFile(new URL('../.env', import.meta.url), 'utf8').then((txt) => {
    for (const line of txt.split('\n')) {
      const m = line.match(/^([A-Z_]+)=(.*)$/);
      if (m && !process.env[m[1]]) process.env[m[1]] = m[2].trim();
    }
  });
}

export function need(name) {
  const v = process.env[name];
  if (!v) {
    console.error(`Variable manquante dans .env : ${name}`);
    process.exit(1);
  }
  return v;
}

export function makeJobId() {
  return `${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 8)}`;
}

// Layout "un dossier par test" : entrées dans <dir>/input, sorties à la racine
// de <dir>. Compat : sans sous-dossier input/, <dir> = dossier d'entrées et les
// sorties vont dans le dossier courant.
export async function resolveInputDir(dir) {
  const inputSub = path.join(dir, 'input');
  if (await stat(inputSub).then((s) => s.isDirectory()).catch(() => false)) {
    return { photoDir: inputSub, outDir: path.resolve(dir) };
  }
  return { photoDir: dir, outDir: process.cwd() };
}

// Vidéo(s) dans photoDir : extraction de frames vers <photoDir>/.frames.
// Retourne le dossier de frames, ou null si aucune vidéo.
export async function extractFrames(photoDir, { maxFrames, frameWidth }) {
  const entries = await readdir(photoDir);
  const videos = entries.filter((f) => VIDEO_EXTS.has(path.extname(f).toLowerCase()));
  if (videos.length === 0) return null;
  const framesDir = path.join(photoDir, '.frames');
  await rm(framesDir, { recursive: true, force: true });
  await mkdir(framesDir, { recursive: true });
  for (const [vi, v] of videos.entries()) {
    const src = path.join(photoDir, v);
    const dur = parseFloat(
      execFileSync('ffprobe', ['-v', 'error', '-show_entries', 'format=duration', '-of', 'csv=p=0', src], {
        encoding: 'utf8',
      })
    );
    const fps = Math.min(3, Math.max(0.5, maxFrames / videos.length / Math.max(dur, 1)));
    console.log(`Vidéo ${v} (${dur.toFixed(0)}s) → extraction à ${fps.toFixed(2)} img/s …`);
    execFileSync('ffmpeg', [
      '-loglevel', 'error', '-i', src,
      '-vf', `fps=${fps},scale='min(${frameWidth},iw)':-2`,
      '-q:v', '2', path.join(framesDir, `v${vi}_%04d.jpg`),
    ]);
  }
  return framesDir;
}

export async function listImages(dir) {
  return (await readdir(dir)).filter((f) => IMAGE_EXTS.has(path.extname(f).toLowerCase())).sort();
}

// Redimensionne les photos avant upload (MASt3R travaille à 512 px : uploader
// du 4032 px est du transfert pur). sips est natif macOS ; fallback ffmpeg.
export async function resizeImages(srcDir, files, maxDim = 1536) {
  const dst = path.join(srcDir, '.resized');
  await rm(dst, { recursive: true, force: true });
  await mkdir(dst, { recursive: true });
  for (const f of files) {
    const src = path.join(srcDir, f);
    const out = path.join(dst, f.replace(/\.[^.]+$/, '.jpg'));
    try {
      execFileSync('sips', ['-s', 'format', 'jpeg', '-Z', String(maxDim), src, '--out', out], {
        stdio: 'ignore',
      });
    } catch {
      execFileSync('ffmpeg', [
        '-loglevel', 'error', '-i', src,
        '-vf', `scale='min(${maxDim},iw)':-2`, '-q:v', '2', out,
      ]);
    }
    process.stdout.write('.');
  }
  console.log(' redimensionnées');
  return dst;
}

export async function uploadImages(supabase, bucket, prefix, dir, files) {
  for (const f of files) {
    const buf = await readFile(path.join(dir, f));
    const { error } = await supabase.storage
      .from(bucket)
      .upload(`${prefix}/${f}`, buf, {
        contentType: CONTENT_TYPES[path.extname(f).toLowerCase()],
        upsert: true,
      });
    if (error) throw new Error(`Upload ${f} : ${error.message}`);
    process.stdout.write('.');
  }
  console.log(' ok');
}

// Déclenche un job serverless RunPod puis poll jusqu'à COMPLETED.
// Retourne l'output du worker ; throw sur FAILED/CANCELLED/TIMED_OUT ou output.error.
export async function runAndPoll({ endpointId, apiKey, input, pollMs = 10_000, etaHint = '' }) {
  console.log('Déclenchement du job RunPod …');
  const runRes = await fetch(`https://api.runpod.ai/v2/${endpointId}/run`, {
    method: 'POST',
    headers: { Authorization: `Bearer ${apiKey}`, 'Content-Type': 'application/json' },
    body: JSON.stringify({ input }),
  });
  if (!runRes.ok) throw new Error(`RunPod /run : HTTP ${runRes.status} ${await runRes.text()}`);
  const { id: runId } = await runRes.json();
  console.log(`Run ${runId} — polling toutes les ${pollMs / 1000}s${etaHint ? ` (${etaHint})` : ''} …`);

  const t0 = Date.now();
  let output;
  for (;;) {
    await new Promise((r) => setTimeout(r, pollMs));
    const res = await fetch(`https://api.runpod.ai/v2/${endpointId}/status/${runId}`, {
      headers: { Authorization: `Bearer ${apiKey}` },
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
  return output;
}

// Télécharge un objet du bucket, réassemblé depuis .partNNN si parts > 0.
export async function downloadOutput(supabase, bucket, key, parts = 0) {
  if (parts > 0) {
    const chunks = [];
    for (let i = 0; i < parts; i++) {
      const partKey = `${key}.part${String(i).padStart(3, '0')}`;
      const { data, error } = await supabase.storage.from(bucket).download(partKey);
      if (error) throw new Error(`Download ${partKey} : ${error.message}`);
      chunks.push(Buffer.from(await data.arrayBuffer()));
      process.stdout.write(`\rpart ${i + 1}/${parts}   `);
    }
    console.log();
    return Buffer.concat(chunks);
  }
  const { data, error } = await supabase.storage.from(bucket).download(key);
  if (error) throw new Error(`Download ${key} : ${error.message}`);
  return Buffer.from(await data.arrayBuffer());
}
