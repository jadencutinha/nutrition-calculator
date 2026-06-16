import express from 'express';
import dotenv from 'dotenv';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

dotenv.config();

const __dirname = path.dirname(fileURLToPath(import.meta.url));

const API_KEY = process.env.OPENAI_API_KEY;
const MODEL = process.env.OPENAI_MODEL || 'gpt-4o';
const PORT = process.env.PORT || 3000;
const ENDPOINT = 'https://api.openai.com/v1/chat/completions';

if (!API_KEY) {
  console.error('[fatal] OPENAI_API_KEY is not set. Define it in .env before starting.');
  process.exit(1);
}

const app = express();
app.use(express.json({ limit: '25mb' }));
app.use(express.static(path.join(__dirname, 'public')));

// ── Analysis instruction set ────────────────────────────────────────────────
// The model is constrained to a strict JSON contract so the frontend can render
// a deterministic dashboard. All numeric fields are numbers (no units inline).
const SYSTEM_PROMPT = `You are a nutritional analysis engine. Given a single food image, identify the food and produce a rigorous, quantitative nutritional profile.

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
  }
}

Rules:
- caloricBreakdown percentages must sum to ~100.
- Provide 4-8 micronutrient entries ordered by nutritional significance.
- All *_g and *_mg fields are numbers only; do not embed units.
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
        max_tokens: 2000,
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

app.get('/api/health', (_req, res) => {
  res.json({ status: 'ok', model: MODEL });
});

app.listen(PORT, () => {
  console.log(`Nutritional analysis engine listening on http://localhost:${PORT}`);
  console.log(`Model: ${MODEL}`);
});
