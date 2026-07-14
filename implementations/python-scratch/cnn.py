# Implementasi Convolutional Layer from Scratch
# Conv2D forward + backward manual pake numpy. Ngga pake framework.

import numpy as np
from itertools import product

# Operasi convolution itu intinya geser-geser kernel (filter) di atas input.
# Setiap posisi, kernel ngelakuin dot product sama region input yang ditempelin.
# Hasilnya: feature map — peta dimana kernel "aktif" di bagian input mana.

class Conv2D:
    def __init__(self, in_channels, out_channels, kernel_size=3, lr=0.01):
        """
        Init weight pake metode He: W ~ N(0, sqrt(2 / (in_ch * k * k))).
        Scaling ini penting biar gradient ngga vanishing/exploding di awal.
        """
        scale = np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.w = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * scale
        self.b = np.zeros((out_channels, 1))

    def forward(self, x):
        """
        x shape: (batch, in_ch, H, W)
        output shape: (batch, out_ch, H_out, W_out)

        convolution = sliding window dot product.
        Biar cepet, pake im2col: ubah input jadi matrix, convolution jadi matrix multiply.
        """
        self.x = x
        batch, in_ch, h, w = x.shape
        out_ch, _, k, _ = self.w.shape
        h_out = h - k + 1
        w_out = w - k + 1

        # im2col: extract semua patch jendela k×k jadi baris-baris matrix
        cols = []
        for i, j in product(range(h_out), range(w_out)):
            patch = x[:, :, i:i+k, j:j+k]  # (batch, in_ch, k, k)
            cols.append(patch.reshape(batch, -1))
        col_matrix = np.stack(cols, axis=2)  # (batch, in_ch*k*k, h_out*w_out)

        # flatten weight
        w_flat = self.w.reshape(out_ch, -1)  # (out_ch, in_ch*k*k)

        # convolution = weight @ col_matrix
        output = w_flat @ col_matrix + self.b.reshape(-1, 1)  # (out_ch, h_out*w_out)
        return output.reshape(batch, out_ch, h_out, w_out)

    def backward(self, grad_output, lr=0.01):
        """
        Backprop convolution: gradient dari output layer di-propagate balik.
        Since convolution linear, gradient-nya convolution juga — bedanya:
        - Forward: filter di-slide di input
        - Backward: flipped filter di-slide di grad_output
        """
        batch, out_ch, h_out, w_out = grad_output.shape
        _, _, k, _ = self.w.shape

        # gradient weight: convolution input sama grad_output
        # dibalik: weight gradient = input ⊛ grad_output
        grad_w = np.zeros_like(self.w)
        for oc in range(out_ch):
            for ic in range(self.w.shape[1]):
                for i, j in product(range(k), range(k)):
                    region = self.x[:, ic, i:i+h_out, j:j+w_out]
                    grad_w[oc, ic, i, j] = np.sum(region * grad_output[:, oc, :, :])

        # gradient bias: jumlahin semua grad_output
        grad_b = np.sum(grad_output, axis=(0, 2, 3), keepdims=True).reshape(out_ch, 1)

        # gradient input: convolution flipped weight sama grad_output
        grad_input = np.zeros_like(self.x)
        w_flipped = np.rot90(self.w, 2, axes=(2, 3))
        for ic in range(self.w.shape[1]):
            for oc in range(out_ch):
                for i, j in product(range(h_out + k - 1), range(w_out + k - 1)):
                    h_start = max(0, i - k + 1); h_end = min(i + 1, h_out)
                    w_start = max(0, j - k + 1); w_end = min(j + 1, w_out)
                    k_h_start = max(0, k - 1 - i); k_h_end = k_h_start + (h_end - h_start)
                    k_w_start = max(0, k - 1 - j); k_w_end = k_w_start + (w_end - w_start)
                    grad_input[:, ic, i, j] += np.sum(
                        grad_output[:, oc, h_start:h_end, w_start:w_end] *
                        w_flipped[oc, ic, k_h_start:k_h_end, k_w_start:k_w_end],
                        axis=(1, 2)
                    )

        self.w -= lr * grad_w / batch
        return grad_input


if __name__ == "__main__":
    # test: input gambar random 1×1@5×5, convolution 1→2 channel, kernel 3×3
    x = np.random.randn(2, 1, 5, 5)
    conv = Conv2D(in_channels=1, out_channels=2, kernel_size=3)
    out = conv.forward(x)
    print(f"input:  {x.shape}")
    print(f"output: {out.shape}  (expected: 2, 2, 3, 3)")
    print("conv2d forward test: OK")
