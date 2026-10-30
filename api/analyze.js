// Vercel serverless function: receives one camera frame and asks Gemini to analyse it.
// The Gemini API key stays on the server (set GEMINI_API_KEY in Vercel → Settings → Environment Variables).
//
// Speed choices:
//  - a fast "lite" model by default, with automatic fallback if a model name is unavailable
//  - low thinking level (thinking adds seconds of latency)
//  - a strict response schema + short output so the model writes less
//  - the browser already shrinks the frame to <=768 px (one image tile = fewest tokens)

const MODELS = [
  process.env.GEMINI_MODEL,
  "gemini-3.5-flash-lite",
  "gemini-2.5-flash-lite",
  "gemini-flash-lite-latest",
  "gemini-flash-latest",
].filter(Boolean);

let workingModel = null;      // remembered while the function instance stays warm
let thinkingSupported = true; // turned off if a model rejects thinkingConfig

const PROMPT = `You analyse one frame from a street camera for a smart-city monitoring system.
Find:
- garbage: litter or waste lying on road, footpath, drain or open ground (plastic, bags, bottles, cans, paper, food waste, piles, overflowing bins)
- littering: a person or vehicle occupant in the act of throwing or dropping waste
- traffic: congestion, illegal or footpath parking, wrong-way driving, riders without helmets, triple riding, blocked road, accident
Rules: report only what is clearly visible. Use short labels (2-4 words). box_2d is [ymin,xmin,ymax,xmax] normalised to 0-1000. At most 8 detections. Summary: one sentence under 25 words. recommended_action: one short instruction for city staff, or "" if nothing is needed.`;

const SCHEMA = {
  type: "OBJECT",
  properties: {
    garbage_detected: { type: "BOOLEAN" },
    littering_detected: { type: "BOOLEAN" },
    traffic_issue_detected: { type: "BOOLEAN" },
    cleanliness_score: { type: "INTEGER" },
    traffic_density: { type: "STRING", enum: ["none", "low", "medium", "high"] },
    vehicle_count: { type: "INTEGER" },
    person_count: { type: "INTEGER" },
    severity: { type: "STRING", enum: ["none", "low", "medium", "high"] },
    summary: { type: "STRING" },
    recommended_action: { type: "STRING" },
    detections: {
      type: "ARRAY",
      items: {
        type: "OBJECT",
        properties: {
          type: { type: "STRING", enum: ["garbage", "littering", "traffic"] },
          label: { type: "STRING" },
          confidence: { type: "NUMBER" },
          box_2d: { type: "ARRAY", items: { type: "INTEGER" } },
        },
        required: ["type", "label", "box_2d"],
      },
    },
  },
  required: [
    "garbage_detected", "littering_detected", "traffic_issue_detected", "cleanliness_score",
    "traffic_density", "vehicle_count", "person_count", "severity", "summary", "detections",
  ],
};

const FOCUS = {
  garbage: "\nFocus on garbage and littering first; still fill every field.",
  traffic: "\nFocus on traffic first; still fill every field.",
};

async function callGemini(model, apiKey, data, mode, useThinking) {
  const generationConfig = {
    temperature: 0.1,
    maxOutputTokens: 900,
    responseMimeType: "application/json",
    responseSchema: SCHEMA,
  };
  if (useThinking) generationConfig.thinkingConfig = { thinkingLevel: "low" };

  const ctrl = new AbortController();
  const timer = setTimeout(() => ctrl.abort(), 25000);
  try {
    const r = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${model}:generateContent`,
      {
        method: "POST",
        signal: ctrl.signal,
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify({
          contents: [{
            role: "user",
            parts: [
              { inline_data: { mime_type: "image/jpeg", data } },
              { text: PROMPT + (FOCUS[mode] || "") },
            ],
          }],
          generationConfig,
        }),
      }
    );
    const body = await r.json().catch(() => ({}));
    return { ok: r.ok, status: r.status, body };
  } finally {
    clearTimeout(timer);
  }
}

function clean(result) {
  const clamp = (n, lo, hi) => Math.min(hi, Math.max(lo, Math.round(Number(n) || 0)));
  result.cleanliness_score = clamp(result.cleanliness_score, 0, 100);
  result.vehicle_count = clamp(result.vehicle_count, 0, 999);
  result.person_count = clamp(result.person_count, 0, 999);
  result.detections = (Array.isArray(result.detections) ? result.detections : [])
    .filter((d) => Array.isArray(d.box_2d) && d.box_2d.length === 4)
    .slice(0, 8)
    .map((d) => ({ ...d, box_2d: d.box_2d.map((v) => clamp(v, 0, 1000)) }));
  return result;
}

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST" });
  }
  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) return res.status(500).json({ error: "GEMINI_API_KEY is not set on the server." });

  const { image, mode } = req.body || {};
  if (!image || typeof image !== "string") return res.status(400).json({ error: "Missing image." });
  const data = image.replace(/^data:image\/\w+;base64,/, "");
  if (data.length > 3_000_000) return res.status(413).json({ error: "Image too large." });

  const started = Date.now();
  const queue = workingModel ? [workingModel, ...MODELS.filter((m) => m !== workingModel)] : MODELS;
  let lastError = "No model available.";
  let lastStatus = 502;

  try {
    for (const model of queue) {
      let out = await callGemini(model, apiKey, data, mode, thinkingSupported);

      // Older models reject thinkingLevel: retry once without it.
      if (!out.ok && out.status === 400 && /think/i.test(out.body?.error?.message || "")) {
        thinkingSupported = false;
        out = await callGemini(model, apiKey, data, mode, false);
      }

      if (out.ok) {
        const text = out.body?.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";
        let result;
        try {
          result = JSON.parse(text.replace(/^```json\s*|```$/g, "").trim());
        } catch {
          lastError = "The AI reply could not be read. Try again.";
          continue;
        }
        workingModel = model;
        result = clean(result);
        result.model = model;
        result.ms = Date.now() - started;
        return res.status(200).json(result);
      }

      lastStatus = out.status;
      lastError = out.body?.error?.message || `Gemini error ${out.status}`;
      // Only fall through to the next model when this model name is unavailable.
      const modelMissing = out.status === 404 || /not found|not supported|unknown model/i.test(lastError);
      if (!modelMissing) break;
    }
  } catch (err) {
    const aborted = err.name === "AbortError";
    return res.status(aborted ? 504 : 500).json({ error: aborted ? "Gemini took too long. Try again." : err.message });
  }

  const status = lastStatus === 429 ? 429 : lastStatus === 400 || lastStatus === 403 ? lastStatus : 502;
  return res.status(status).json({ error: lastError });
}

export const config = { api: { bodyParser: { sizeLimit: "3mb" } } };
