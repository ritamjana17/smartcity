/*
 * CityEye street cleanliness model (runs in the browser).
 *
 * Trained in training/cityeye_training.ipynb on the Kaggle "Clean/Dirty Road Classification" dataset.
 *   features = MobileNetV2 (ImageNet) embedding of the image, 1,280 numbers
 *   p(dirty) = sigmoid(weights · features + bias)        weights/bias: model/cleanliness_model.json
 */
(function () {
  const TF = "https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@4.22.0/dist/tf.min.js";
  const MOBILENET = "https://cdn.jsdelivr.net/npm/@tensorflow-models/mobilenet@2.1.1/dist/mobilenet.min.js";
  const MODEL_URL = "/model/cleanliness_model.json";

  const scripts = {};
  function loadScript(src) {
    if (src === TF && window.tf) return Promise.resolve();
    if (!scripts[src]) {
      scripts[src] = new Promise((ok, fail) => {
        const s = document.createElement("script");
        s.src = src; s.async = true; s.crossOrigin = "anonymous";
        s.onload = ok;
        s.onerror = () => { delete scripts[src]; fail(new Error("Could not download " + src.split("/npm/")[1])); };
        document.head.appendChild(s);
      });
    }
    return scripts[src];
  }

  let loading = null;
  async function load(url = MODEL_URL) {
    if (!loading) {
      loading = (async () => {
        const r = await fetch(url, { cache: "no-cache" });
        if (!r.ok) throw Object.assign(new Error("model/cleanliness_model.json was not found. Run the Kaggle notebook and add its output to the project."), { missing: true });
        const cfg = await r.json();
        await loadScript(TF);
        await loadScript(MOBILENET);
        const net = await window.mobilenet.load({ version: 2, alpha: 1.0 });
        const weights = window.tf.tensor1d(cfg.weights);
        // check that the browser features have the same size as the trained weights
        const probe = window.tf.tidy(() => net.infer(window.tf.zeros([224, 224, 3]), true));
        const size = probe.shape[probe.shape.length - 1]; probe.dispose();
        if (size !== cfg.weights.length) throw new Error(`Feature size mismatch: browser ${size}, model ${cfg.weights.length}`);
        return { cfg, net, weights, bias: cfg.bias };
      })();
      loading.catch(() => { loading = null; });
    }
    return loading;
  }

  // el: <video>, <img> or <canvas>. Returns probability (0-1) that the street is dirty.
  async function predict(el) {
    const m = await load();
    const tf = window.tf;
    const p = tf.tidy(() => {
      const emb = m.net.infer(el, true).reshape([-1]);
      return tf.sigmoid(tf.add(tf.sum(tf.mul(emb, m.weights)), m.bias));
    });
    const v = (await p.data())[0]; p.dispose();
    return v;
  }

  window.CityEyeModel = { load, predict, loadScript, TF, MODEL_URL, info: () => loading };
})();
