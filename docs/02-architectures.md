# Arsitektur Neural Network

> FNN, CNN, RNN, LSTM, Transformer — kapan pake yang mana dan kenapa.

---

## 1. Feedforward Neural Network (FNN) / MLP

FNN adalah arsitektur paling dasar. Data mengalir satu arah: input → hidden layer(s) → output. Ngga ada looping, ngga ada memory.

**Struktur:**
- Input layer: flat vector (misal 784 untuk MNIST 28×28)
- Hidden layer: fully connected + activation
- Output layer: tergantung task (sigmoid buat binary, softmax buat multiclass, linear buat regression)

**Matematika:**
```
z_1 = W_1 · x + b_1
a_1 = ReLU(z_1)
z_2 = W_2 · a_1 + b_2
y = softmax(z_2)
```

**Kelemahan:** 
- Ngga bisa handle spatial structure (setiap pixel diperlakukan independen)
- Ngga bisa handle sequential data
- Parameternya gede banget buat high-dimensional input

**Cocok buat:** Tabular data, regression, classification dengan fitur yang udah di-feature-engineer.

---

## 2. Convolutional Neural Network (CNN)

CNN dirancang khusus buat data dengan spatial structure — gambar, video, bahkan audio spectrogram. Bedanya sama FNN: setiap neuron cuma terhubung ke sebagian kecil input (local connectivity), bukan seluruh input.

### 2.1 Convolution Operation

Convolution itu operasi geser-geser filter (kernel) kecil di atas input:

```
(f ∗ g)[i, j] = Σ_m Σ_n f[m, n] × g[i + m, j + n]
```

Inti-nya: filter kecil (3×3, 5×5) belajar mendeteksi pola lokal — edge, texture, corner. Stacking convolution layers bikin network bisa belajar fitur hierarkis: edge → shape → object part → object.

### 2.2 Komponen Utama CNN

| Komponen | Fungsi | Parameter |
|----------|--------|-----------|
| **Convolution layer** | Deteksi fitur lokal | Kernel size, stride, padding, channels |
| **Activation (ReLU)** | Non-linearity | - |
| **Pooling (Max/Avg)** | Reduksi dimensi, translation invariance | Pool size, stride |
| **Fully connected** | Klasifikasi | Number of neurons |

**Weight sharing:** Filter yang sama digeser-geser di seluruh input. Ini beda krusial sama FNN — jumlah parameter jauh lebih sedikit.

### 2.3 Arsitektur CNN Populer

| Arsitektur | Tahun | Ciri Khas |
|------------|-------|-----------|
| LeNet-5 | 1998 | CNN pertama buat digit recognition |
| AlexNet | 2012 | Deep learning revolution — 8 layers |
| VGGNet | 2014 | Very deep — 16-19 layers, 3×3 filters doang |
| ResNet | 2015 | Skip connection — bisa sampe 152 layers |
| EfficientNet | 2019 | Neural architecture search, optimal scaling |

### 2.4 Kapan Pake CNN?

- Input punya spatial structure (2D grid kayak gambar)
- Pola yang mau dideteksi bersifat lokal di awal, global di akhir
- Invarian terhadap translasi (object dipindah dikit masih dikenali)

Jangan pake CNN buat: tabular data, sequence panjang, atau data tanpa struktur spasial.

---

## 3. Recurrent Neural Network (RNN)

RNN dirancang buat sequential data — time series, teks, audio, video frame. Bedanya: neuron punya hidden state yang ngebawa informasi dari step sebelumnya.

### 3.1 Mekanisme

```
h_t = tanh(W_hh · h_{t-1} + W_xh · x_t + b_h)
y_t = W_hy · h_t + b_y
```

`h_t` itu hidden state di time step t — ngebawa informasi dari step 1 sampai step t. Tapi makin panjang sequencenya, makin susah RNN ngebawa informasi dari awal (vanishing gradient).

### 3.2 Masalah RNN

**Vanishing gradient (long-term dependency):** Waktu backpropagation lewat banyak time step (BPTT — Backpropagation Through Time), gradient bisa mengecil exponensial. Network cuma bisa belajar dependency pendek.

**Exploding gradient:** Sebaliknya — gradient membesar, weight jadi NaN. Diatasi dengan gradient clipping.

### 3.3 Variasi RNN

| Varian | Bedanya |
|--------|---------|
| Bidirectional RNN | Bacanya dari kiri ke kanan + kanan ke kiri, context lebih lengkap |
| Deep RNN | Stacking RNN layers |

---

## 4. Long Short-Term Memory (LSTM)

LSTM adalah solusi buat vanishing gradient di RNN. Ditemukan oleh Hochreiter & Schmidhuber (1997). Intinya: **tambah mekanisme gate yang ngontrol aliran informasi**.

### 4.1 Struktur LSTM Cell

LSTM punya 3 gate:

| Gate | Fungsi | Formula |
|------|--------|---------|
| **Forget gate** | Apa yang mau dilupain dari memory | `f_t = σ(W_f · [h_{t-1}, x_t])` |
| **Input gate** | Info baru apa yang mau disimpen | `i_t = σ(W_i · [h_{t-1}, x_t])` |
| **Output gate** | Apa yang mau dikeluarin dari memory | `o_t = σ(W_o · [h_{t-1}, x_t])` |

Plus candidate memory:
```
c̃_t = tanh(W_c · [h_{t-1}, x_t])
```

Update cell state:
```
c_t = f_t × c_{t-1} + i_t × c̃_t
h_t = o_t × tanh(c_t)
```

**Intuisinya:** Forget gate mutusin apa yang mau dihapus dari memory. Input gate mutusin info baru apa yang ditambahin. Cell state (`c_t`) bawa informasi lintas step dengan perubahan linear — inilah kenapa vanishing gradient berkurang drastis.

### 4.2 LSTM vs RNN

| Aspek | RNN | LSTM |
|-------|-----|------|
| Memory | Hidden state doang | Hidden state + cell state |
| Long-term dependency | Susah | Bisa |
| Parameter count | Lebih dikit | 4× lebih banyak (W_f, W_i, W_o, W_c) |
| Compute | Lebih cepet | Lebih lambat |
| Overfitting | Lebih tahan | Gampang overfit |

### 4.3 GRU (Gated Recurrent Unit)

GRU adalah penyederhanaan LSTM — cuma 2 gate (reset gate, update gate), ngga punya cell state terpisah. Lebih cepet dari LSTM, performanya sering sama.

---

## 5. Transformer

Transformer dari paper "Attention Is All You Need" (Vaswani et al., 2017) adalah arsitektur yang sekarang dominan di NLP dan mulai ngalahin CNN di vision juga.

### 5.1 Self-Attention

Inti transformer: **self-attention mechanism**. Setiap token bisa "memperhatikan" semua token lain dalam satu sequence sekaligus — ngga sequential kayak RNN.

```
Attention(Q, K, V) = softmax(Q × K^T / √d_k) × V
```

- `Q` (Query): "Apa yang saya cari?"
- `K` (Key): "Apa yang saya tawarkan?"
- `V` (Value): "Apa yang saya berikan kalo dicocokin?"

Dot product Q×K^T ngukur similarity antara setiap pasangan token. Dibagi √d_k biar gradient stabil. Softmax ngasih attention weight.

### 5.2 Multi-Head Attention

Daripada satu attention, transformer pake 8-16 head parallel:

```
MultiHead(Q, K, V) = Concat(head_1, ..., head_h) × W_O
head_i = Attention(Q × W_Q_i, K × W_K_i, V × W_V_i)
```

Setiap head bisa belajar "attention" yang berbeda — satu fokus ke grammar, satu ke semantic relationship, satu ke posisi.

### 5.3 Positional Encoding

Self-attention ngga tau urutan token — "saya makan nasi" sama "nasi makan saya" hasil attention-nya sama. Makanya butuh positional encoding:

```
PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
```

### 5.4 Transformer vs RNN/LSTM

| Aspek | RNN/LSTM | Transformer |
|-------|----------|-------------|
| Processing | Sequential | Parallel |
| Long-range dependency | Susah (RNN), bisa (LSTM) | Alami |
| Training speed | Lambat | Cepet (parallel) |
| O(n) compute | O(n) step | O(n²) attention |
| Memory | O(n) | O(n²) |
| Positional info | Built-in (sequential) | Butuh positional encoding |

Transformer unggul di training speed dan long-range dependency. Tapi O(n²) bikin berat buat sequence sangat panjang.

### 5.5 Arsitektur Transformer

**Encoder:** Self-attention → Feedforward → Add&Norm (residual connection + layer norm).

**Decoder:** Self-attention (masked) → Cross-attention → Feedforward → Add&Norm.

**Vision Transformer (ViT):** Image dipecah jadi patches (16×16), tiap patch di-flatten jadi token, terus diproses kayak transformer NLP.

---

## 6. Perbandingan Akhir

| Kriteria | FNN | CNN | RNN | LSTM | Transformer |
|----------|-----|-----|-----|------|-------------|
| Data type | Tabular | Grid (image) | Sequence | Sequence | Sequence |
| Parameter efficiency | Jelek | Baik | Sedang | Sedang | Jelek (kecil) |
| Long dependency | N/A | N/A | Jelek | Baik | Sangat baik |
| Parallelization | Ya | Ya | Ngga | Ngga | Ya |
| Training speed | Cepat | Cepat | Lambat | Lambat | Cepat |
| Compute O() | O(n²) | O(k²×n) | O(n) | O(n) | O(n²) |

**Panduan milih arsitektur:**
1. **Image** → CNN (ViT kalo datanya banyak)
2. **Time series pendek** → RNN/GRU
3. **Time series panjang** → LSTM atau Transformer
4. **Text/NLP** → Transformer (BERT, GPT)
5. **Tabular** → FNN atau XGBoost
6. **Data dikit** → CNN/RNN (lebih data-efficient)
7. **Data banyak** → Transformer (scaling terbaik)

---

## Referensi

- LeCun et al. *Gradient-based learning applied to document recognition.* IEEE, 1998.
- Krizhevsky et al. *ImageNet Classification with Deep Convolutional Neural Networks.* NeurIPS, 2012.
- Hochreiter & Schmidhuber. *Long Short-Term Memory.* Neural Computation, 1997.
- Vaswani et al. *Attention Is All You Need.* NeurIPS, 2017.
- Dosovitskiy et al. *An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale.* ICLR, 2021.
