# Fundamental Neural Network

> Neuron, activation function, loss function, forward propagation, backpropagation — dari nol.

---

## 1. Artificial Neuron

Neuron buatan terinspirasi dari biological neuron di otak manusia. Bedanya, biological neuron itu kompleks banget (ada dendrites, axon, synapses, neurotransmitter), sementara neuron buatan cuma melakukan operasi matematika sederhana:

```
input_1 × weight_1 + input_2 × weight_2 + ... + bias → activation → output
```

Secara matematis:

```
z = Σ(wᵢ × xᵢ) + b
a = g(z)
```

Dimana:
- `xᵢ` = input feature
- `wᵢ` = weight (kekuatan koneksi)
- `b` = bias (threshold)
- `g()` = activation function (non-linear)
- `a` = output akhir

**Kenapa butuh bias?** Bias memungkinkan neuron mengaktifkan diri bahkan ketika semua input-nya 0. Bayangin kayak threshold: tanpa bias, neuron cuma bisa bereaksi ke input, ngga punya "default state".

### Biologi vs Komputasi

| Biologi | Komputasi |
|---------|-----------|
| Dendrites | Input features |
| Synapse | Weights |
| Nucleus | Summation + activation |
| Axon | Output |
| Neurotransmitter | Activation signal |

---

## 2. Activation Function

Activation function gunanya **introducing non-linearity**. Kalau cuma operasi linear (weight × input + bias), stacking layer manapun hasilnya tetep linear — setara dengan satu layer doang. Non-linearity bikin network bisa belajar pola kompleks.

### 2.1 Sigmoid

```
σ(z) = 1 / (1 + e^(-z))
Range: (0, 1)
```

**Ciri khas:** Output di antara 0 dan 1 — cocok buat probabilitas.

**Masalah:** 
- Vanishing gradient — di ujung kurva (z besar positif/negatif), gradient mendekati 0, neuron berhenti belajar.
- Bukan zero-centered — output selalu positif, bikin gradient update zigzag.

Dulu populer banget, sekarang udah jarang dipake di hidden layer. Paling sering dipake di output layer binary classification.

### 2.2 Tanh (Hyperbolic Tangent)

```
tanh(z) = (e^z - e^(-z)) / (e^z + e^(-z))
Range: (-1, 1)
```

**Ciri khas:** Zero-centered — output bisa negatif, gradient update lebih stabil daripada sigmoid.

**Masalah:** Vanishing gradient masih terjadi.

Tanh biasanya lebih bagus daripada sigmoid buat hidden layer, tapi sekarang udah kalah sama ReLU.

### 2.3 ReLU (Rectified Linear Unit)

```
ReLU(z) = max(0, z)
Range: [0, ∞)
```

**Ciri khas:** Sederhana, murah komputasi, ngga vanishing gradient untuk nilai positif.

**Masalah:** Dying ReLU — kalo neuron masuk region negatif, gradient-nya 0, neuron bakal mati permanen.

ReLU adalah activation function paling populer dan default buat hidden layer di deep learning. Simple, work, efficient.

### 2.4 Leaky ReLU / PReLU / ELU

Varian ReLU buat ngatasin dying ReLU:

```
Leaky ReLU(z) = max(0.01z, z)
PReLU(z) = max(αz, z)  # α dipelajari
ELU(z) = z if z > 0, else α(e^z - 1)
```

Leaky ReLU ngasih slope kecil 0.01 buat nilai negatif — biar neuron tetep bisa belajar walau di region negatif. PReLU bikin α jadi parameter yang dipelajari.

### 2.5 Softmax

```
softmax(z_i) = e^(z_i) / Σ e^(z_j)
```

Bukan activation function per-neuron, tapi per-layer. Output totalnya pasti 1 — jadi cocok buat multiclass classification probability distribution.

### Tabel Perbandingan

| Fungsi | Range | Pro | Kontra |
|--------|-------|-----|--------|
| Sigmoid | (0, 1) | Output probabilitas | Vanishing gradient, not zero-centered |
| Tanh | (-1, 1) | Zero-centered | Vanishing gradient |
| ReLU | [0, ∞) | Murah, no vanishing gradient | Dying ReLU |
| Leaky ReLU | (-∞, ∞) | Fixes dying ReLU | Parameter tambahan |
| Softmax | (0,1) sum=1 | Probabilitas multiclass | Cuma di output layer |

---

## 3. Loss Function

Loss function mengukur seberapa jauh prediksi network dari target sebenarnya. Training = minimize loss.

### 3.1 Mean Squared Error (MSE)

```
MSE = (1/n) × Σ(y_pred - y_true)²
```

**Cocok buat:** Regression task (prediksi nilai kontinu).

Gradient-nya linear: ∂MSE/∂y_pred = (2/n) × (y_pred - y_true). Makin jauh error, makin besar koreksi.

### 3.2 Binary Cross-Entropy (BCE)

```
BCE = -(1/n) × Σ(y_true × log(y_pred) + (1 - y_true) × log(1 - y_pred))
```

**Cocok buat:** Binary classification.

Kenapa pake log? Karena kita pengen ngasih penalty besar kalo model yakin tapi salah. Misal y_true=1 tapi model output 0.01 → loss-nya gede banget.

### 3.3 Categorical Cross-Entropy (CCE)

```
CCE = -(1/n) × Σ Σ y_true_ij × log(y_pred_ij)
```

**Cocok buat:** Multiclass classification (dengan softmax di output).

### 3.4 Hinge Loss (SVM-style)

```
Hinge = (1/n) × Σ max(0, 1 - y_true × y_pred)
```

**Cocok buat:** Binary classification dengan maksimum margin.

---

## 4. Forward Propagation

Forward pass adalah proses ngirim data dari input layer ke output layer. Setiap layer melakukan:

1. Hitung weighted sum: `z = W × a_prev + b`
2. Apply activation: `a = g(z)`
3. Kirim ke layer berikutnya

Untuk network dengan L layer:

```
a[0] = x                           # input
z[l] = W[l] × a[l-1] + b[l]       # l = 1, 2, ..., L
a[l] = g[l](z[l])                  # activation
y_pred = a[L]                      # output
```

**Dimensi matriks:**
- W[l]: (n_neurons_l, n_neurons_l-1)
- b[l]: (n_neurons_l, 1)
- z[l], a[l]: (n_neurons_l, 1)

Forward pass itu deterministic — untuk input dan weight yang sama, outputnya selalu sama. Yang bikin network belajar adalah proses backward pass.

---

## 5. Backpropagation

Backpropagation adalah algoritma yang bikin deep learning feasible. Ditemukan oleh Rumelhart, Hinton, dan Williams di 1986. Intinya: **compute gradient loss terhadap setiap weight pake chain rule**, biar kita tau cara update weight biar loss turun.

### 5.1 Intuisi

Misal kita punya 3 layer: input → hidden → output. Error di output disebabkan oleh:
- Output weights: kontribusi langsung
- Hidden weights: kontribusi lewat output layer

Chain rule bilang: kalo kita mau tau ∂Loss/∂W_hidden, kita hitung dulu ∂Loss/∂a_output, terus ∂a_output/∂z_hidden, terus ∂z_hidden/∂W_hidden. Dikalikan semua.

### 5.2 Step-by-step

**Step 1: Forward pass**
```
z[1] = W[1] × x + b[1]
a[1] = g(z[1])
z[2] = W[2] × a[1] + b[2]
a[2] = g(z[2])
loss = L(a[2], y)
```

**Step 2: Output layer gradient**
```
δ[2] = ∂loss/∂z[2] = ∂loss/∂a[2] × ∂a[2]/∂z[2]
∂loss/∂W[2] = δ[2] × a[1]ᵀ
∂loss/∂b[2] = δ[2]
```

**Step 3: Hidden layer gradient (backpropagation)**
```
δ[1] = (W[2]ᵀ × δ[2]) × ∂a[1]/∂z[1]
∂loss/∂W[1] = δ[1] × xᵀ
∂loss/∂b[1] = δ[1]
```

**Step 4: Weight update (gradient descent)**
```
W = W - learning_rate × ∂loss/∂W
b = b - learning_rate × ∂loss/∂b
```

### 5.3 Kenapa Backpropagation Efisien?

Naive approach: setiap weight dihitung gradient-nya dengan numerical differentiation — O(W²) komputasi, ngga scalable.

Backpropagation: O(W) — sekali forward, sekali backward. Ini yang bikin network dengan jutaan weight bisa ditrain.

### 5.4 General Case

Untuk layer ke-l secara umum:

```
δ[l] = (W[l+1]ᵀ × δ[l+1]) × g'(z[l])
∂loss/∂W[l] = δ[l] × a[l-1]ᵀ
∂loss/∂b[l] = δ[l]
```

Ini pattern yang sama terus dari layer terakhir sampai pertama — makanya disebut *back*propagation.

---

## 6. Initialization

Bobot awal network ngaruh besar ke convergence:

- **Zero initialization:** Semua neuron belajar hal yang sama (symmetry problem). Network ngga pernah belajar.
- **Random small:** `W ~ N(0, 0.01)` — works buat small network.
- **Xavier/Glorot:** `W ~ U(-√(6/n_in+n_out), √(6/n_in+n_out))` — recommended buat tanh/sigmoid.
- **He:** `W ~ N(0, √(2/n_in))` — recommended buat ReLU.

Bias biasanya di-init 0.

---

## Referensi

- Rumelhart, Hinton, Williams. *Learning representations by back-propagating errors.* Nature, 1986.
- Glorot, Bengio. *Understanding the difficulty of training deep feedforward neural networks.* AISTATS, 2010.
- He et al. *Delving deep into rectifiers.* ICCV, 2015.
