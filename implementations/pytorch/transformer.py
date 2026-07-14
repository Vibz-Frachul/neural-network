# Transformer Encoder dari Scratch — PyTorch
# Implementasi arsitektur dari paper "Attention Is All You Need" (Vaswani et al., 2017).
# Ini versi encoder-only, cocok buat klasifikasi atau feature extraction.

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Inti transformer ada di self-attention: setiap token bisa "ngintip" semua token lain.
# Dibanding RNN yang baca sequential, transformer paralel — makanya training-nya cepet.
# Tapi O(n²) memory, jadi kalo sequence panjang (>512) bisa berat.

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_k = d_model // n_heads
        self.n_heads = n_heads
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)

    def forward(self, x, mask=None):
        batch, seq_len, d_model = x.shape
        Q = self.w_q(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        K = self.w_k(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        V = self.w_v(x).view(batch, seq_len, self.n_heads, self.d_k).transpose(1, 2)

        # Attention(Q,K,V) = softmax(QK^T / √d_k) V
        scores = Q @ K.transpose(-2, -1) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn = F.softmax(scores, dim=-1)
        out = (attn @ V).transpose(1, 2).contiguous().view(batch, seq_len, d_model)
        return self.w_o(out)


class FeedForward(nn.Module):
    def __init__(self, d_model=64, d_ff=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class TransformerEncoder(nn.Module):
    def __init__(self, vocab_size, d_model=64, n_heads=4, d_ff=256, num_layers=3, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_enc = PositionalEncoding(d_model)
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'attention': MultiHeadAttention(d_model, n_heads),
                'ff': FeedForward(d_model, d_ff),
                'norm1': nn.LayerNorm(d_model),
                'norm2': nn.LayerNorm(d_model)
            })
            for _ in range(num_layers)
        ])
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x, mask=None):
        x = self.embedding(x) + self.pos_enc(x)
        for layer in self.layers:
            attn_out = layer['attention'](layer['norm1'](x), mask)
            x = x + attn_out                          # residual connection
            ff_out = layer['ff'](layer['norm2'](x))
            x = x + ff_out
        return self.classifier(x.mean(dim=1))          # global average pooling


class PositionalEncoding(nn.Module):
    """Positional encoding pake sin/cos — biar transformer tau urutan token."""
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return self.pe[:, :x.size(1), :]


if __name__ == "__main__":
    import numpy as np
    np.random.seed(42)

    # dummy data: 100 sample, tiap sample 32 token, vocabulary 1000
    x = torch.randint(0, 1000, (16, 32))
    y = torch.randint(0, 4, (16,))

    model = TransformerEncoder(vocab_size=1000, d_model=64, n_heads=4, num_layers=3, num_classes=4)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(20):
        optimizer.zero_grad()
        loss = F.cross_entropy(model(x), y)
        loss.backward()
        optimizer.step()
        acc = (model(x).argmax(dim=1) == y).float().mean().item()
        if epoch % 5 == 0:
            print(f"epoch {epoch:2d} | loss: {loss.item():.4f} | acc: {acc:.4f}")

    print("\ntransformer training dummy test: OK")
