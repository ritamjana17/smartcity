// Vercel serverless function: receives a camera frame, asks Gemini to analyse it.
// The Gemini API key stays on the server (set GEMINI_API_KEY in Vercel settings).

const MODEL = process.env.GEMINI_MODEL || "gemini-flash-latest";

const PROMPT = `You are an AI smart-city surveillance analyst looking at one frame from a street camera.
Detect:
1. GARBAGE / LITTER lying on the street, sidewalk, drains or open ground (plastic, bags, bottles, cans, paper, food waste, piles, overflowing bins).
2. LITTERING ACTS: a person or vehicle occupant in the act of throwing/dropping waste.
3. TRAFFIC issues: congestion, illegal/wrong parking, wrong-way driving, signal jumping, riders without helmets, triple riding, blocked road, accident.

Return ONLY JSON in this exact shape:
{
  "garbage_detected": boolean,
  "littering_detected": boolean,
  "traffic_issue_detected": boolean,
  "cleanliness_score": integer 0-100 (100 = perfectly clean),
  "traffic_density": "none" | "low" | "medium" | "high",
  "vehicle_count": integer,
  "person_count": integer,
  "severity": "none" | "low" | "medium" | "high",
  "summary": "one short sentence describing the scene",
  "detections": [
    {
      "type": "garbage" | "littering" | "traffic",
      "label": "short label e.g. plastic bottles, person throwing wrapper, car parked on footpath",
      "confidence": number 0-1,
      "box_2d": [ymin, xmin, ymax, xmax]  // integers normalised 0-1000
    }
  ],
  "recommended_action": "short action for city staff, or empty string"
}
Only report things you can actually see. If nothing is found, return empty detections and false flags.`;

export default async function handler(req, res) {
  if (req.method !== "POST") {
    res.setHeader("Allow", "POST");
    return res.status(405).json({ error: "Use POST" });
  }

  const apiKey = process.env.GEMINI_API_KEY;
  if (!apiKey) {
    return res.status(500).json({ error: "GEMINI_API_KEY is not set on the server." });
  }

  try {
    const { image, mode } = req.body || {};
    if (!image || typeof image !== "string") {
      return res.status(400).json({ error: "Missing image (base64 JPEG)." });
    }
    const data = image.replace(/^data:image\/\w+;base64,/, "");
    if (data.length > 4_000_000) {
      return res.status(413).json({ error: "Image too large." });
    }

    let focus = "";
    if (mode === "garbage") focus = "\nFocus mainly on garbage and littering; still fill every field.";
    if (mode === "traffic") focus = "\nFocus mainly on traffic; still fill every field.";

    const gRes = await fetch(
      `https://generativelanguage.googleapis.com/v1beta/models/${MODEL}:generateContent`,
      {
        method: "POST",
        headers: { "Content-Type": "application/json", "x-goog-api-key": apiKey },
        body: JSON.stringify({
          contents: [
            {
              role: "user",
              parts: [
                { inline_data: { mime_type: "image/jpeg", data } },
                { text: PROMPT + focus },
              ],
            },
          ],
          generationConfig: {
            temperature: 0.2,
            responseMimeType: "application/json",
          },
        }),
      }
    );

    const body = await gRes.json();
    if (!gRes.ok) {
      const msg = body?.error?.message || `Gemini error ${gRes.status}`;
      return res.status(gRes.status === 429 ? 429 : 502).json({ error: msg });
    }

    const text =
      body?.candidates?.[0]?.content?.parts?.map((p) => p.text || "").join("") || "";
    let result;
    try {
      result = JSON.parse(text.replace(/^```json\s*|```$/g, "").trim());
    } catch {
      return res.status(502).json({ error: "Could not read AI response.", raw: text.slice(0, 500) });
    }

    result.detections = Array.isArray(result.detections) ? result.detections : [];
    result.model = MODEL;
    return res.status(200).json(result);
  } catch (err) {
    return res.status(500).json({ error: err.message || "Server error" });
  }
}

export const config = {
  api: { bodyParser: { sizeLimit: "4mb" } },
};
