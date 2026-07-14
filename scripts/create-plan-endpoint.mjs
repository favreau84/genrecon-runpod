#!/usr/bin/env node
// Crée le template + l'endpoint serverless RunPod pour le worker Plane-DUSt3R,
// puis affiche la ligne RUNPOD_PLAN_ENDPOINT_ID à ajouter au .env.
// Idempotent : réutilise le template/endpoint s'ils existent déjà.

import { loadEnv, need } from '../lib/common.mjs';

const API = 'https://rest.runpod.io/v1';
const TEMPLATE_NAME = 'planedust3r-worker';
const ENDPOINT_NAME = 'planedust3r-plan';
const IMAGE = 'ghcr.io/favreau84/planedust3r-worker:latest';
const NETWORK_VOLUME_ID = 'rzh0jzujtx'; // genrecon-weights (EU-RO-1)
// A40 d'abord : 48 Go comme la L40S mais ~2× moins cher (0,44 $/h secure)
const GPU_TYPE_IDS = ['NVIDIA A40', 'NVIDIA L40S'];

async function api(method, path, body) {
  const res = await fetch(`${API}${path}`, {
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

  let template = (await api('GET', '/templates')).find((t) => t.name === TEMPLATE_NAME);
  if (template) {
    console.log(`Template existant réutilisé : ${template.id}`);
  } else {
    template = await api('POST', '/templates', {
      name: TEMPLATE_NAME,
      imageName: IMAGE,
      containerDiskInGb: 25,
      env: {
        SUPABASE_URL: need('SUPABASE_URL'),
        SUPABASE_SECRET_KEY: need('SUPABASE_SECRET_KEY'),
        HF_TOKEN: need('HF_TOKEN'),
      },
    });
    console.log(`Template créé : ${template.id}`);
  }

  let endpoint = (await api('GET', '/endpoints')).find((e) => e.name === ENDPOINT_NAME);
  if (endpoint) {
    console.log(`Endpoint existant réutilisé : ${endpoint.id}`);
  } else {
    endpoint = await api('POST', '/endpoints', {
      name: ENDPOINT_NAME,
      templateId: template.id,
      gpuTypeIds: GPU_TYPE_IDS,
      workersMin: 0,
      workersMax: 1,
      idleTimeout: 5,
      executionTimeoutMs: 900_000,
      flashboot: true,
      networkVolumeId: NETWORK_VOLUME_ID,
      scalerType: 'QUEUE_DELAY',
      scalerValue: 4,
    });
    console.log(`Endpoint créé : ${endpoint.id}`);
  }

  console.log(`\nÀ ajouter au .env si absent :\nRUNPOD_PLAN_ENDPOINT_ID=${endpoint.id}`);
}

main().catch((e) => {
  console.error('❌', e.message ?? e);
  process.exit(1);
});
