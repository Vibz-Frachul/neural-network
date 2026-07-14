# Training Neural Network

> Optimizer, regularization, hyperparameter — cara bikin neural network beneran belajar.

---

## 1. Gradient Descent

Gradient descent adalah algoritma optimasi inti. Idenya: ikutin arah turunan (negative gradient) dari loss function buat nyari weight optimal.

```
θ = θ - η × ∇_θ J(θ)
```

Dimana θ = weight, η = learning rate, J = loss function.

### 1.1 Variasi Gradient Descent

| Varian | Data per update | Kelebihan | Kekurangan |
|--------|----------------|-----------|------------|
| **Batch GD** | Seluruh dataset | Stabil, deterministik | Lambat, ngga muat memory |
| **Stochastic GD** | 1 sample | Cepet per step, bisa escape local minima | Noisy, ngga stabil convergence |
| **Mini-batch GD** | Batch (16-512) | Balance — paling umum dipake | - |

### 1.2 Learning Rate

Learning rate adalah hyperparameter paling krusial.

- **Terlalu kecil:** Convergence lambat banget, bisa stuck di local minima.
- **Terlalu besar:** Divergence — loss naik terus, weight oscillation.
- **Learning rate schedule:** Turunin LR seiring training. Kayak `StepLR`, `CosineAnnealing`, `ReduceLROnPlateau`.

**Golden range:** Biasanya mulai dari 1e-3 (Adam) atau 1e-2 (SGD dengan momentum).

---

## 2. Optimizer

Optimizer adalah implementasi gradient descent dengan tambahan adaptive learning rate dan momentum.

### 2.1 SGD with Momentum

Momentum: kalo gradient terus ke arah yang sama, velocity numpuk. Kalo gradient berubah arah, velocity teredam.

```
v_t = β × v_{t-1} + (1-β) × ∇_θ J(θ)
θ = θ - η × v_t
```

β biasanya 0.9. Bayangin kayak bola yang menggelinding turun — ngga langsung berhenti kalo ketemu bump kecil.

### 2.2 Adam (Adaptive Moment Estimation)

Adam = Momentum + RMSProp. Paling populer, works out of the box untuk banyak task.

```
m_t = β₁ × m_{t-1} + (1-β₁) × g_t          # first moment (mean)
v_t = β₂ × v_{t-1} + (1-β₂) × g_t²         # second moment (variance)
m̂_t = m_t / (1 - β₁^t)                       # bias correction
v̂_t = v_t / (1 - β₂^t)
θ = θ - η × m̂_t / (√v̂_t + ε)
```

**Kenapa Adam works:** Setiap parameter punya learning rate sendiri-sendiri berdasarkan seberapa sering dia diupdate. Parameter yang jarang dapet gradient → learning rate lebih gede.

### 2.3 Perbandingan Optimizer

| Optimizer | Adaptive LR | Momentum | Kapan Pake |
|-----------|-------------|----------|------------|
| SGD | Ngga | Ngga | Simple baseline |
| SGD + Momentum | Ngga | Ya | Computer vision, generalization bagus |
| Adam | Ya | Ya | Default — NLP, speech, umum |
| AdamW | Ya | Ya | Adam + weight decay terpisah, lebih bagus |
| RMSProp | Ya | Ngga | RNN, online learning |

**Saran:** Start dengan Adam, learning rate 3e-4 atau 1e-3. Kalo mau generalization lebih bagus, coba SGD + Momentum setelah dapet feel.

---

## 3. Regularization

Regularization tujuannya biar network ngga overfit — ngga ngapalin training data tapi gagal di data baru.

### 3.1 L1 / L2 Regularization (Weight Decay)

Tambah penalty ke loss berdasarkan besar weight.

```
L2: J = original_loss + λ × Σ w²
L1: J = original_loss + λ × Σ |w|
```

- **L2:** Weight mengecil (tapi ngga nol). Smooth regularization.
- **L1:** Weight bisa jadi nol — sparse solution. Cocok buat feature selection.

**Weight decay:** Istilah lain buat L2 regularization di optimizer kayak AdamW.

### 3.2 Dropout

Selama training, neuron "dimatiin" secara random dengan probability p. Di test time, semua neuron aktif, output dikali (1-p).

**Kenapa work:** Mencegah ko-adaptasi antar neuron — setiap neuron harus belajar fitur yang berguna sendiri, bukan bergantung ke neuron lain.

**Dropout rate:** Biasanya 0.2-0.5. Makin besar network, makin tinggi dropout.

### 3.3 Batch Normalization

Normalisasi output layer: mean=0, variance=1, terus di-scale/shift pake parameter yang dipelajari.

```
μ_B = mean(x)
σ_B² = var(x)
x̂ = (x - μ_B) / √(σ_B² + ε)
y = γ × x̂ + β
```

**Kenapa work:**
- Stabil distribusi (ngurangin internal covariate shift)
- Gradient flow lebih lancar
- Ada efek regularisasi ringan
- Bisa pake learning rate lebih gede

Batch norm biasanya ditaro sebelum atau sesudah activation function.

### 3.4 Early Stopping

Stop training kalo validation loss mulai naik (nggak turun dalam N epoch). Sederhana tapi efektif.

### 3.5 Data Augmentation

Bikin variasi training data secara artifisial: rotasi, flip, crop, noise, color jitter. Ini regularization paling bagus buat computer vision.

---

## 4. Hyperparameter Tuning

| Hyperparameter | Range | Efek |
|----------------|-------|------|
| Learning rate | 1e-5 — 1e-1 | Paling krusial |
| Batch size | 16 — 512 | Kecil → noisy, besar → stabil |
| Epochs | 10 — 100+ | Early stopping nentuin |
| Hidden layers | 1 — 5+ | Depth |
| Hidden units | 64 — 1024 | Width |
| Dropout rate | 0.0 — 0.5 | Regularization strength |
| Weight decay | 1e-5 — 1e-2 | L2 strength |

**Strategi tuning:**
1. **Grid search:** Coba semua kombinasi — mahal.
2. **Random search:** Sampling random — lebih efisien.
3. **Bayesian optimization:** Pake history buat predict kombinasi bagus (Optuna, Hyperopt).

---

## 5. Masalah Umum & Solusi

| Masalah | Gejala | Solusi |
|---------|--------|--------|
| Overfitting | Train loss turun, val loss naik | Regularization, data augmentation, smaller network |
| Underfitting | Train loss tinggi | Bigger network, kurangi regularization, training lebih lama |
| Vanishing gradient | Network ngga belajar | ReLU, batch norm, residual connection, weight init baik |
| Exploding gradient | Loss jadi NaN | Gradient clipping, lower LR |
| Dying ReLU | Banyak neuron output 0 terus | Leaky ReLU, PReLU |
| Class imbalance | Akurasi tinggi tapi recall rendah | Weighted loss, oversampling, Focal Loss |

---

## Referensi

- Kingma, Ba. *Adam: A Method for Stochastic Optimization.* ICLR, 2015.
- Ioffe, Szegedy. *Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift.* ICML, 2015.
- Srivastava et al. *Dropout: A Simple Way to Prevent Neural Networks from Overfitting.* JMLR, 2014.
- Loshchilov, Hutter. *Decoupled Weight Decay Regularization (AdamW).* ICLR, 2019.
