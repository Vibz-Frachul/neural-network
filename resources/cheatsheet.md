# Cheatsheet Neural Network

> Formula-formula kunci yang sering dipake — referensi cepet.

---

## Neuron

```
z = W · x + b
a = g(z)
```

| Simbol | Arti |
|--------|------|
| x | Input |
| W | Weight matrix |
| b | Bias vector |
| z | Pre-activation (weighted sum) |
| a | Activation (output neuron) |
| g() | Activation function |

---

## Activation Functions

| Fungsi | Formula | Range | Turunan |
|--------|---------|-------|---------|
| Sigmoid | `1/(1+e⁻ᶻ)` | (0, 1) | `σ(z)(1-σ(z))` |
| Tanh | `(eᶻ−e⁻ᶻ)/(eᶻ+e⁻ᶻ)` | (-1, 1) | `1 - tanh²(z)` |
| ReLU | `max(0, z)` | [0, ∞) | `1(z>0)` |
| Leaky ReLU | `max(0.01z, z)` | (-∞, ∞) | `0.01(z≤0) + 1(z>0)` |
| Softmax | `eᶻⁱ/Σeᶻʲ` | (0, 1) sum=1 | `σᵢ(δᵢⱼ − σⱼ)` |

---

## Forward Propagation

```
z[1] = W[1] · x + b[1]
a[1] = g[1](z[1])
z[2] = W[2] · a[1] + b[2]
a[2] = g[2](z[2])
⋮
y_pred = a[L]
```

---

## Backpropagation

```
δ[L] = ∇a_J ⊙ g'(z[L])              # output layer
δ[l] = (W[l+1]ᵀ · δ[l+1]) ⊙ g'(z[l])  # hidden layer
∂J/∂W[l] = δ[l] · a[l-1]ᵀ
∂J/∂b[l] = δ[l]
```

---

## Common Loss Functions

| Loss | Formula | Task |
|------|---------|------|
| MSE | `(1/n)Σ(ŷ−y)²` | Regression |
| MAE | `(1/n)Σ|ŷ−y|` | Regression (outlier robust) |
| BCE | `−Σ(ylog ŷ + (1−y)log(1−ŷ))` | Binary classification |
| CCE | `−ΣΣ y_ij log(ŷ_ij)` | Multi-class classification |
| Hinge | `Σ max(0, 1−y·ŷ)` | SVM binary |

---

## Optimizer Update Rules

**SGD:** `θ = θ − η · ∇J(θ)`

**SGD + Momentum:**
```
v = β·v + (1−β)·∇J(θ)
θ = θ − η·v
```

**Adam:**
```
m = β₁·m + (1−β₁)·g
v = β₂·v + (1−β₂)·g²
θ = θ − η · m̂ / (√v̂ + ε)
```

---

## Convolution

```
output_size = (input_size − kernel_size + 2·padding) / stride + 1

Num params = (kernel_h × kernel_w × in_ch + 1) × out_ch
```

---

## Self-Attention

```
Attention(Q, K, V) = softmax(Q · Kᵀ / √d_k) · V

Q/K/V = projection of input with learned weights
d_k = dimension per attention head
```

---

## Parameter Count

| Layer | Parameter Count |
|-------|----------------|
| Linear(in, out) | `in × out + out` |
| Conv2D(in, out, k) | `k² × in × out + out` |
| LSTM(in, h) | `4 × (in × h + h² + h)` |
| Multi-Head Attn(d, h) | `4d²` (approx) |

---

## Weight Initialization

| Init | Distribution | For Activation |
|------|-------------|----------------|
| Xavier/Glorot | `U(−√(6/(nᵢₙ+nₒᵤₜ)), √(6/(nᵢₙ+nₒᵤₜ)))` | Tanh, Sigmoid |
| He | `N(0, √(2/nᵢₙ))` | ReLU, Leaky ReLU |
| Default PyTorch | `U(−√(1/nᵢₙ), √(1/nᵢₙ))` | - |

---

## Evaluation Metrics

| Metric | Formula | Task |
|--------|---------|------|
| Accuracy | `(TP+TN)/(TP+TN+FP+FN)` | Classification |
| Precision | `TP/(TP+FP)` | Classification |
| Recall | `TP/(TP+FN)` | Classification |
| F1 | `2·P·R/(P+R)` | Classification |
| MSE | `(1/n)Σ(ŷ−y)²` | Regression |
| R² | `1 − SS_res/SS_tot` | Regression |

---

## Tips Cepet

| Masalah | Cek |
|---------|-----|
| Loss ngga turun | LR terlalu kecil/gede, weight init salah |
| Loss NaN | Exploding gradient → gradient clipping |
| Overfit | Regularization, data augmentation |
| Underfit | Model terlalu kecil, training kurang |
| Training lambat | Batch size terlalu kecil, no GPU |
