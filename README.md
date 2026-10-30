# CityEye – AI-Powered Smart City Surveillance

Camera-first web app for phones and desktops. Point the camera at a street and CityEye finds
**garbage on the street**, **people littering** and **traffic problems** using Google Gemini.
It also counts vehicles and people live on the device.

## Features
- Opens straight to the camera (rear camera on phones, webcam on laptops)
- **Scan** mode: tap the shutter, the frame freezes and the result appears in about 1–3 s
- **Auto watch** mode: scans every 3–30 s on its own, with a beep and vibration when something is found
- Live object boxes on the device (TensorFlow.js COCO-SSD): vehicles, people and litter-like objects, no waiting
- AI result: verdict, cleanliness score, traffic density, vehicle and people counts, suggested action, labelled boxes
- History of incidents with annotated snapshots, optional GPS map links, CSV export
- Upload a photo or video when no camera is available
- Installable with "Add to Home Screen"; light and dark mode
- Desktop: two-column layout, Space bar scans

## Why it is fast
- Frames are shrunk to 768 px before upload (one Gemini image tile)
- Default model is a fast lite model (`gemini-3.5-flash-lite`), with automatic fallback to other models
- Low thinking level, strict JSON schema and short answers
- The server function is warmed up while you aim, and runs in Mumbai (`bom1`, see `vercel.json`)

## Project structure
```
index.html        whole app UI (camera, live detection, results, history)
api/analyze.js    Vercel serverless function that calls Gemini (key stays secret)
vercel.json       region + function settings
manifest.json     makes it installable on phones
icon.svg
```

## Deploy
1. Get a Gemini key at https://aistudio.google.com/apikey
2. Push this folder to GitHub.
3. On Vercel: Add New → Project → import the repo → Framework preset **Other**.
4. Settings → Environment Variables → add `GEMINI_API_KEY` = your key.
   Optional: `GEMINI_MODEL` (for example `gemini-3.6-flash` for higher accuracy, a bit slower).
5. Deploy, then open the `https://…vercel.app` link on your phone and tap **Start camera**.

If you change the key later: edit it in Vercel, then Deployments → ⋯ → Redeploy.

## Notes
- The camera only works over HTTPS (Vercel gives you this) or on localhost.
- On the Gemini free tier, use an Auto watch interval of 10 s or more.
- History is stored only on the device (localStorage).
- Recording people in public may be regulated where you live; check local rules before real deployment.
