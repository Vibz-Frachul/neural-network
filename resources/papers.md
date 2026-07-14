# Paper Penting Neural Network

> Daftar paper foundational yang wajib dibaca — lengkap dengan rangkuman dan kenapa paper itu penting.

---

## Foundational Papers

### 1. A Logical Calculus of the Ideas Immanent in Nervous Activity
- **Penulis:** McCulloch & Pitts (1943)
- **Arsitektur:** — (teori)
- **Inti:** Model matematis pertama dari neuron buatan. Ngelakuin operasi logika threshold. Dasar dari semua neural network.
- **Kenapa penting:** Paper yang melahirkan bidang ini.

### 2. The Perceptron: A Perceiving and Recognizing Automaton
- **Penulis:** Rosenblatt (1958)
- **Arsitektur:** Perceptron
- **Inti:** Algoritma pertama yang bisa belajar dari data secara otomatis. Perceptron meng-update weight berdasarkan error prediksi.
- **Kenapa penting:** Bukti pertama bahwa mesin bisa belajar dari contoh.

### 3. Perceptrons
- **Penulis:** Minsky & Papert (1969)
- **Arsitektur:** Perceptron
- **Inti:** Bukti matematis bahwa single-layer perceptron ngga bisa solve XOR. Bikin funding AI kering ("AI Winter" pertama).
- **Kenapa penting:** Nyadarin komunitas butuh multi-layer network.

### 4. Learning representations by back-propagating errors
- **Penulis:** Rumelhart, Hinton & Williams (1986)
- **Arsitektur:** MLP + Backpropagation
- **Inti:** Algoritma backpropagation yang efisien buat train multi-layer network. Chain rule buat compute gradient dari output ke input.
- **Kenapa penting:** Algoritma yang bikin deep learning feasible. Tanpa ini, ngga ada modern neural network.

### 5. Gradient-based learning applied to document recognition
- **Penulis:** LeCun et al. (1998)
- **Arsitektur:** CNN (LeNet-5)
- **Inti:** Arsitektur CNN pertama yang berhasil — convolution + pooling + fully connected. Dipake buat check recognition.
- **Kenapa penting:** Fondasi dari semua arsitektur CNN modern.

---

## Deep Learning Revolution

### 6. ImageNet Classification with Deep Convolutional Neural Networks
- **Penulis:** Krizhevsky, Sutskever & Hinton (2012) — **AlexNet**
- **Arsitektur:** CNN (AlexNet)
- **Inti:** Deep CNN (8 layer) menang telak ImageNet 2012. GPU training + ReLU + Dropout + Data Augmentation.
- **Kenapa penting:** Momen "big bang" deep learning. Ngebuktiin deep networks works di skala besar.

### 7. Very Deep Convolutional Networks for Large-Scale Image Recognition
- **Penulis:** Simonyan & Zisserman (2014) — **VGGNet**
- **Arsitektur:** CNN (VGGNet)
- **Inti:** Pake 3×3 convolution doang, stacking 16-19 layers. Sederhana tapi efektif.
- **Kenapa penting:** Nunjukkin bahwa depth (kedalaman) itu penting.

### 8. Going Deeper with Convolutions
- **Penulis:** Szegedy et al. (2014) — **GoogLeNet / Inception**
- **Arsitektur:** CNN (Inception)
- **Inti:** Inception module: convolution parallel berbagai ukuran (1×1, 3×3, 5×5) dalam satu layer.
- **Kenapa penting:** Inovasi arsitektur — ngga cuma stacking, tapi parallel processing.

### 9. Deep Residual Learning for Image Recognition
- **Penulis:** He et al. (2015) — **ResNet**
- **Arsitektur:** CNN (ResNet)
- **Inti:** Skip connection / residual connection: `output = F(x) + x`. Bikin training network >100 layer jadi feasible.
- **Kenapa penting:** Ngatasin vanishing gradient di network sangat dalam. Sekarang residual connection ada di mana-mana.

---

## Sequence Modeling

### 10. Long Short-Term Memory
- **Penulis:** Hochreiter & Schmidhuber (1997)
- **Arsitektur:** LSTM
- **Inti:** Mekanisme gate (input, forget, output) buat ngontrol informasi di memory cell. Ngatasin vanishing gradient di RNN.
- **Kenapa penting:** Standar de facto buat sequence modeling sebelum transformer.

### 11. Empirical Evaluation of Gated Recurrent Neural Networks on Sequence Modeling
- **Penulis:** Chung et al. (2014)
- **Arsitektur:** GRU
- **Inti:** Penyederhanaan LSTM — cum 2 gate, tanpa cell state. Performa similar, lebih cepet.
- **Kenapa penting:** Alternatif ringan dari LSTM.

---

## Transformer Era

### 12. Attention Is All You Need
- **Penulis:** Vaswani et al. (2017)
- **Arsitektur:** Transformer
- **Inti:** Self-attention mechanism. Ngga pake recurrence sama sekali. Parallel processing, lebih cepet dari RNN.
- **Kenapa penting:** Arsitektur paling berpengaruh setelah CNN. Dasar dari GPT, BERT, semua LLM modern.

### 13. BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding
- **Penulis:** Devlin et al. (2018)
- **Arsitektur:** Transformer (Encoder)
- **Inti:** Bidirectional pre-training: Masked Language Model + Next Sentence Prediction.
- **Kenapa penting:** Paradigma pre-training + fine-tuning buat NLP. Transfer learning untuk teks.

### 14. An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale
- **Penulis:** Dosovitskiy et al. (2020) — **ViT**
- **Arsitektur:** Vision Transformer (ViT)
- **Inti:** Image dipecah jadi patches 16×16, diproses kayak token transformer. Ngalahin CNN kalo data besar.
- **Kenapa penting:** Transformer dominance extends ke computer vision.

---

## Training & Optimization

### 15. Adam: A Method for Stochastic Optimization
- **Penulis:** Kingma & Ba (2014)
- **Topik:** Optimizer
- **Inti:** Adaptive learning rate: momentum + RMSProp. Works out of the box buat banyak task.
- **Kenapa penting:** Optimizer paling populer. Default choice buat training.

### 16. Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift
- **Penulis:** Ioffe & Szegedy (2015)
- **Topik:** Regularization
- **Inti:** Normalisasi output layer: stabil distribusi, gradient flow lebih lancar, regularization effect.
- **Kenapa penting:** Teknik yang bikin training deep network jauh lebih stabil.

### 17. Dropout: A Simple Way to Prevent Neural Networks from Overfitting
- **Penulis:** Srivastava et al. (2014)
- **Topik:** Regularization
- **Inti:** Randomly drop neuron selama training, cegah ko-adaptasi.
- **Kenapa penting:** Regularization technique yang sederhana tapi efektif.

---

## Cara Baca Paper

1. **Baca judul + abstract** — Pahamin inti.
2. **Baca introduction** — Kenapa paper ini ditulis.
3. **Liat gambar** — Biasanya arsitektur ada di figure 1-3.
4. **Baca method** — Gimana caranya.
5. **Baca experiments** — Apakah beneran works?
6. **Baca conclusion** — Kesimpulan + future work.
7. **Skip related work** — Kalo ngga butuh konteks sejarah.
