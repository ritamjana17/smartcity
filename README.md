# CityEye – AI-Powered Smart City Surveillance

Mobile web app that uses the phone camera (or an uploaded video/photo) to detect
**garbage on streets**, **people littering**, and **traffic problems**, using Google Gemini.

## Features
- Responsive design: thumb-friendly bottom bar on phones, two-column console on desktop (laptop webcam works too)
- Desktop shortcuts: Space = check frame, M = start/stop monitoring
- Light and dark mode follow the device setting
- Live rear/front camera on phones (HTTPS required — Vercel provides it)
- Upload a street video or photo instead of using the camera
- Modes: All / Garbage / Traffic
- Bounding boxes drawn on detected garbage, littering, and traffic issues
- Cleanliness score, traffic density, vehicle and people counts
- Auto-monitor every 3–30 s, with sound + vibration alerts
- Incident log with snapshots, optional GPS map link, CSV export
- "Add to Home Screen" works like an app (PWA manifest)

## Project structure
```
index.html        ← the whole phone UI (camera, overlay, log)
api/analyze.js    ← Vercel serverless function that calls Gemini (key stays secret)
manifest.json     ← makes it installable on phones
icon.svg
package.json
```

## 1. Get a Gemini API key
Go to https://aistudio.google.com/apikey and create a key.

## 2. Push to GitHub
```bash
cd smartcity
git init
git add .
git commit -m "CityEye smart city surveillance"
git branch -M main
git remote add origin https://github.com/<your-username>/cityeye.git
git push -u origin main
```
Never commit your API key. `.env` is already in `.gitignore`.

## 3. Deploy on Vercel
1. vercel.com → **Add New → Project** → import the GitHub repo.
2. Framework preset: **Other**. Leave build settings empty.
3. **Environment Variables** → add `GEMINI_API_KEY` = your key.
   (Optional: `GEMINI_MODEL`, default is `gemini-flash-latest`, e.g. `gemini-3.6-flash`.)
4. Click **Deploy**. Open the `https://…vercel.app` link on your phone.

If you add the key after the first deploy, go to Deployments → ⋯ → **Redeploy**.

## 4. Use on the phone
- Tap **Start camera** and allow camera access.
- Tap **Analyze frame** for one check, or **Start auto-monitor** for continuous checks.
- Browser menu → **Add to Home Screen** to install it like an app.

## Run locally (optional)
```bash
npm i -g vercel
cp .env.example .env    # put your key in it
vercel dev              # open http://localhost:3000
```

## Notes
- Each analysis is one Gemini request; a 5 s interval ≈ 12 requests/minute.
  On the free tier, use 10–30 s if you hit "quota exceeded".
- Frames are resized to 960 px before upload to stay fast and cheap.
- The incident log is stored only on the phone (localStorage).
- Recording people in public may be regulated where you live; check local rules before real deployment.
