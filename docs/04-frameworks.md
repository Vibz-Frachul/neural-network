# Framework Perbandingan

> PyTorch vs TensorFlow vs JAX — kapan pake yang mana.

---

## 1. PyTorch

**Developer:** Meta AI (Facebook)
**Rilis:** 2016
**Paradigma:** Define-by-run (dynamic computation graph)

### Kelebihan
- **Debugging mudah:** Graph dibangun runtime, bisa pake debugger Python biasa.
- **Pythonic:** Ngoding kayak nulis Python biasa, ngga ada session/placeholder kayak TF1.
- **Penelitian:** Dominan di academic research, sebagian besar paper baru pake PyTorch.
- **Community:** Ekosistem gede (Hugging Face, Lightning, torchvision, torchaudio).
- **TorchScript:** Bisa export ke production (via `torch.jit.trace` atau `torch.export`).

### Kekurangan
- **Production deployment:** Kurang mature dibanding TF Serving/TensorRT. Tapi makin baik.
- **Mobile:** Dukungan terbatas (PyTorch Mobile ada tapi belum sebagus TF Lite).
- **Static graph optimization:** Kalah sama JAX dalam beberapa benchmark.

### Kapan Pake PyTorch
- Lo bikin riset / prototype
- Lo bikin model NLP pake Hugging Face
- Lo mau debugging mudah
- Lo males ribet sama session, placeholder, graph building explicit

---

## 2. TensorFlow / Keras

**Developer:** Google
**Rilis:** 2015 (TF1), 2019 (TF2 with eager execution)
**Paradigma:** Define-and-run (TF1), eager + graph (TF2)

### Kelebihan
- **Production:** TF Serving, TF Lite, TF.js, TPU — ekosistem production paling mature.
- **Keras API:** High-level API yang simpel banget.
- **TensorBoard:** Visualization tools yang mature.
- **TPU support:** Kalo mau training di TPU Google, TF pilihan utama.
- **Ecosystem:** Luas — dari mobile sampe web.

### Kekurangan
- **API complexity:** Banyak layer API (tf.keras, tf.Module, tf.function).
- **Debugging:** Lebih susah daripada PyTorch terutama kalo pake @tf.function graph mode.
- **Swift changes:** API sering berubah (dari TF1 ke TF2 banyak breaking changes).

### Kapan Pake TensorFlow
- Lo deploy model ke production (mobile, web, server)
- Lo pake TPU
- Lo perlu visualization super detail (TensorBoard)
- Lo bikin aplikasi mobile/embedded

---

## 3. JAX

**Developer:** Google Research
**Rilis:** 2018
**Paradigma:** Functional, composable transformations

### Kelebihan
- **XLA compilation:** Just-in-time compile ke GPU/TPU — kenceng.
- **Functional purity:** Ngga ada hidden state — fungsi pure.
- **Autograd (`grad`):** Gradien bisa di-compose (`grad(grad(fn))` untuk second derivative).
- **`vmap`:** Vectorization otomatis — tulis code buat satu sample, `vmap` bikin batch.
- **`pmap`:** Parallelism di banyak device.
- **Flexibilitas:** Lo bisa nulis research yang eksperimental banget.

### Kekurangan
- **Ecosystem:** Masih kecil. Flax, Haiku, Equinox — semuanya third-party.
- **Learning curve:** Functional programming style, beda sama PyTorch/TF.
- **Debugging:** Error messages kadang cryptic.
- **Community:** Lebih kecil — dokumentasi kurang lengkap.

### Kapan Pake JAX
- Lo jago functional programming
- Lo butuh performa maksimal (XLA + custom training loops)
- Lo bikin research yang perlu gradien orde tinggi
- Lo nyaman dokumentasi serba kurang

---

## 4. Perbandingan Langsung

| Kriteria | PyTorch | TensorFlow | JAX |
|----------|---------|------------|-----|
| **Ease of use** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Debugging** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **Research** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Production** | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Performance** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Mobile** | ⭐⭐ | ⭐⭐⭐⭐ | ⭐ |
| **TPU support** | ⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **Community** | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐ |
| **Ecosystem** | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐ |
| **Learning curve** | Rendah | Sedang | Tinggi |

---

## 5. Rekomendasi

| Situasi | Pilih |
|---------|-------|
| Lo baru belajar deep learning | **PyTorch** |
| Lo bikin research / paper | **PyTorch** |
| Lo pake Hugging Face | **PyTorch** |
| Lo pake NLP / LLM | **PyTorch** |
| Lo deploy ke Android | **TensorFlow Lite** |
| Lo deploy ke browser | **TensorFlow.js** |
| Lo pake TPU Google | **TensorFlow** atau **JAX** |
| Lo butuh performa maksimal | **JAX** |
| Lo production industrial scale | **TensorFlow** |
| Lo eksperimen functional programming | **JAX** |

**Bottom line:** Kalo bingung, pake PyTorch. 80% use case terpenuhi.

---

## Referensi

- *PyTorch documentation.* https://pytorch.org/docs/stable/
- *TensorFlow documentation.* https://www.tensorflow.org/learn
- *JAX documentation.* https://jax.readthedocs.io/en/latest/
