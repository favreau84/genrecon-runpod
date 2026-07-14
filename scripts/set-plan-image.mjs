#!/usr/bin/env node
// Épingle l'image du template serverless planedust3r-worker sur un tag précis.
// Usage : node scripts/set-plan-image.mjs [tag]   (défaut : HEAD du repo local)
//
// Nécessaire après chaque build : RunPod réutilise l'image ":latest" déjà
// présente en cache sur la machine du worker — seul un changement de tag
// force le pull de la nouvelle image.

import { execFileSync } from 'node:child_process';
import { loadEnv, need } from '../lib/common.mjs';

const TEMPLATE_NAME = 'planedust3r-worker';
const IMAGE_BASE = 'ghcr.io/favreau84/planedust3r-worker';

async function api(method, path, body) {
  const res = await fetch(`https://rest.runpod.io/v1${path}`, {
    method,
    headers: {
      Authorization: `Bearer ${need('RUNPOD_API_KEY')}`,
      'Content-Type': 'application/json',
    },
    body: body ? JSON.stringify(body) : undefined,
  });
  const text = await res.text();
  if (!res.ok) throw new Error(`${method} ${path} : HTTP ${res.status} ${text}`);
  return text ? JSON.parse(text) : null;
}

async function main() {
  await loadEnv();
  const tag = process.argv[2] ?? execFileSync('git', ['rev-parse', 'HEAD'], { encoding: 'utf8' }).trim();
  const image = `${IMAGE_BASE}:${tag}`;

  const template = (await api('GET', '/templates')).find((t) => t.name === TEMPLATE_NAME);
  if (!template) throw new Error(`Template ${TEMPLATE_NAME} introuvable — lancer create-plan-endpoint.mjs`);
  if (template.imageName === image) {
    console.log(`Template déjà sur ${image}`);
    return;
  }
  await api('PATCH', `/templates/${template.id}`, { imageName: image });
  console.log(`Template ${template.id} → ${image}`);
}

main().catch((e) => {
  console.error('❌', e.message ?? e);
  process.exit(1);
});
