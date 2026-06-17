import express from 'express';
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import { mkdir, readFile, writeFile } from 'node:fs/promises';
import { randomUUID } from 'node:crypto';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL || 'gpt-4o';
const PORT = process.env.PORT || 3000;
const ENDPOINT = 'https://api.openai.com/v1/chat/completions';

const DATA_DIR = path.join(__dirname, 'data');
const LOG_FILE = path.join(DATA_DIR, 'log.json');

if (!API_KEY) {
  console.error('[fatal] OPENAI_API_KEY is not set. Define it in .env before starting.');
  process.exit(1);
}

const app = express();
app.use(express.json({ limit: '25mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Persistence (single-file JSON store) ────────────────────────────────────
// Writes are serialized through a promise chain so concurrent requests can't
// interleave a read-modify-write and clobber each other.
let writeChain = Promise.resolve();

async function readLog() {
  try {
    const raw = await readFile(LOG_FILE, 'utf8');
    const parsed = JSON.parse(raw);
    return Array.isArray(parsed) ? parsed : [];
  } catch (err) {
    if (err.code === 'ENOENT') return [];
    throw err;
  }
}

function mutateLog(fn) {
  // Chain the mutation so writes happen one at a time.
  const next = writeChain.then(async () => {
    const entries = await readLog();
    const result = await fn(entries);
    await writeFile(LOG_FILE, JSON.stringify(entries, null, 2), 'utf8');
    return result;
  });
  // Keep the chain alive even if this mutation rejects.
  writeChain = next.catch(() => {});
  return next;
}

const isDateString = (s) => typeof s === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(s);
const toNumber = (v) => {
  const n = Number(v);
  return Number.isFinite(n) ? n : null;
};

// ── Analysis instruction set ────────────────────────────────────────────────
// The model is constrained to a strict JSON contract so the frontend can render
// a deterministic dashboard. All numeric fields are numbers (no units inline).
const SYSTEM_PROMPT = `You are a nutritional analysis engine. Given a single food image, identify the food and produce a rigorous, quantitative nutritional profile plus one recipe suggestion.

Estimate values for the portion visible in the image. When a value cannot be determined, use your best evidence-based estimate rather than null. Return ONLY a single valid JSON object matching this exact schema — no markdown, no code fences, no commentary:

{
  "identification": {
    "name": "specific descriptive name of the dish",
    "category": "food group / cuisine classification",
    "confidence": 0-100,
    "components": ["distinct food items detected in the image"]
  },
  "serving": {
    "description": "human-readable portion, e.g. '1 bowl' or '2 slices'",
    "estimatedWeightGrams": number
  },
  "energy": {
    "calories": number,
    "caloriesPer100g": number,
    "kilojoules": number
  },
  "macronutrients": {
    "protein_g": number,
    "totalCarbohydrate_g": number,
    "dietaryFiber_g": number,
    "sugars_g": number,
    "addedSugars_g": number,
    "totalFat_g": number,
    "saturatedFat_g": number,
    "transFat_g": number,
    "cholesterol_mg": number,
    "sodium_mg": number,
    "potassium_mg": number
  },
  "caloricBreakdown": {
    "proteinPct": number,
    "carbPct": number,
    "fatPct": number
  },
  "micronutrients": [
    { "name": "Vitamin C", "amount": "mg or mcg with unit", "dailyValuePct": number }
  ],
  "glycemic": {
    "index": number,
    "load": number,
    "classification": "low | medium | high"
  },
  "dietary": {
    "flags": ["e.g. vegetarian, vegan, gluten-free, keto-friendly, high-protein"],
    "allergens": ["e.g. dairy, gluten, nuts, soy, eggs, shellfish"]
  },
  "assessment": {
    "healthScore": 0-100,
    "summary": "one to two sentence expert assessment",
    "notes": ["concise factual observations relevant to nutrition or diet"]
  },
  "recipe": {
    "title": "name of a simple recipe that prominently features this food",
    "description": "one to two sentence description of the dish",
    "servings": number,
    "prepTimeMinutes": number,
    "cookTimeMinutes": number,
    "ingredients": ["quantity + ingredient, e.g. '2 ripe bananas'"],
    "steps": ["clear, ordered preparation instruction"],
    "tips": ["optional short serving or substitution tip"]
  }
}

Rules:
- caloricBreakdown percentages must sum to ~100.
- Provide 4-8 micronutrient entries ordered by nutritional significance.
- All *_g and *_mg fields are numbers only; do not embed units.
- The recipe must prominently feature the identified food, stay approachable (8 or fewer ingredients), and include 3-7 steps.
- If no food is present in the image, set identification.name to "unknown", confidence to 0, numeric fields to 0, and omit a meaningful recipe (use empty strings/arrays).
- Output the raw JSON object and nothing else.`;

app.post('/api/analyze', async (req, res) => {
  const { image, mimeType } = req.body || {};

  if (!image) {
    return res.status(400).json({ error: 'No image data received.' });
  }

  const dataUrl = `data:${mimeType || 'image/jpeg'};base64,${image}`;

  try {
    const upstream = await fetch(ENDPOINT, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
        Authorization: `Bearer ${API_KEY}`
      },
      body: JSON.stringify({
        model: MODEL,
        max_tokens: 3000,
        temperature: 0.2,
        response_format: { type: 'json_object' },
        messages: [
          { role: 'system', content: SYSTEM_PROMPT },
          {
            role: 'user',
            content: [
              { type: 'text', text: 'Analyze the food in this image and return the nutritional profile JSON.' },
              { type: 'image_url', image_url: { url: dataUrl, detail: 'high' } }
            ]
          }
        ]
      })
    });

    if (!upstream.ok) {
      let detail = `Upstream API error ${upstream.status}`;
      try {
        const errBody = await upstream.json();
        detail = errBody.error?.message || detail;
      } catch {
        /* non-JSON error body */
      }
      console.error('[analyze] upstream failure:', detail);
      return res.status(502).json({ error: detail });
    }

    const payload = await upstream.json();
    const content = payload.choices?.[0]?.message?.content;

    if (!content) {
      return res.status(502).json({ error: 'Empty response from analysis model.' });
    }

    let parsed;
    try {
      parsed = JSON.parse(content);
    } catch {
      // Fallback: extract the first JSON object if the model added stray text.
      const match = content.match(/\{[\s\S]*\}/);
      if (!match) {
        return res.status(502).json({ error: 'Model returned unparseable output.' });
      }
      parsed = JSON.parse(match[0]);
    }

    parsed._meta = { model: MODEL, generatedAt: new Date().toISOString() };
    return res.json(parsed);
  } catch (err) {
    console.error('[analyze] unexpected error:', err);
    return res.status(500).json({ error: 'Internal analysis error. Check server logs.' });
  }
});

// ── Daily intake log ─────────────────────────────────────────────────────────
app.get('/api/log', async (req, res) => {
  try {
    const { from, to } = req.query;
    let entries = await readLog();
    if (isDateString(from)) entries = entries.filter((e) => e.date >= from);
    if (isDateString(to)) entries = entries.filter((e) => e.date <= to);
    entries.sort((a, b) => (a.date === b.date ? a.createdAt.localeCompare(b.createdAt) : a.date.localeCompare(b.date)));
    res.json({ entries });
  } catch (err) {
    console.error('[log:get] error:', err);
    res.status(500).json({ error: 'Could not read the intake log.' });
  }
});

app.post('/api/log', async (req, res) => {
  const body = req.body || {};
  const date = body.date;
  const name = typeof body.name === 'string' ? body.name.trim() : '';
  const calories = toNumber(body.calories);

  if (!isDateString(date)) return res.status(400).json({ error: 'A valid date (YYYY-MM-DD) is required.' });
  if (!name) return res.status(400).json({ error: 'A food name is required.' });
  if (calories === null || calories < 0) return res.status(400).json({ error: 'Calories must be a non-negative number.' });

  const entry = {
    id: randomUUID(),
    date,
    name,
    calories,
    protein_g: toNumber(body.protein_g),
    carb_g: toNumber(body.carb_g),
    fat_g: toNumber(body.fat_g),
    source: body.source === 'analysis' ? 'analysis' : 'manual',
    createdAt: new Date().toISOString()
  };

  try {
    await mutateLog((entries) => { entries.push(entry); });
    res.status(201).json({ entry });
  } catch (err) {
    console.error('[log:post] error:', err);
    res.status(500).json({ error: 'Could not save the entry.' });
  }
});

app.delete('/api/log/:id', async (req, res) => {
  const { id } = req.params;
  try {
    let removed = false;
    await mutateLog((entries) => {
      const idx = entries.findIndex((e) => e.id === id);
      if (idx !== -1) { entries.splice(idx, 1); removed = true; }
    });
    if (!removed) return res.status(404).json({ error: 'Entry not found.' });
    res.json({ ok: true });
  } catch (err) {
    console.error('[log:delete] error:', err);
    res.status(500).json({ error: 'Could not delete the entry.' });
  }
});

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', model: MODEL });
});

await mkdir(DATA_DIR, { recursive: true });

app.listen(PORT, () => {
  console.log(`Nutritional analysis engine listening on http://localhost:${PORT}`);
  console.log(`Model: ${MODEL}`);
});
