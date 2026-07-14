# Implementasi Neuron from Scratch

import numpy as np

# Kita bikin neuron paling sederhana: satu neuron, satu input.
# Neuron ini belajar hubungan linear antara input dan output.
# Misalnya: f(x) = 2x + 1 — kita kasih data, neuron belajar sendiri weight dan bias-nya.

class Neuron:
    def __init__(self):
        """Init weight dan bias pake nilai random kecil biar symmetry broken."""
        self.w = np.random.randn() * 0.1
        self.b = np.random.randn() * 0.1

    def forward(self, x):
        """
        Forward pass: z = w * x + b.
        Disini pake linear activation (identity) — ngga ada non-linearity.
        """
        return self.w * x + self.b

    def train(self, xs, ys, lr=0.01, epochs=100):
        """
        Training pake gradient descent manual.
        Loss: MSE = 1/n * Σ(y_pred - y_true)^2
        Gradient: ∂MSE/∂w = 2/n * Σ(y_pred - y_true) * x
                  ∂MSE/∂b = 2/n * Σ(y_pred - y_true)
        """
        for epoch in range(epochs):
            y_preds = np.array([self.forward(x) for x in xs])

            # hitung gradient
            dw = (2.0 / len(xs)) * np.sum((y_preds - ys) * np.array(xs))
            db = (2.0 / len(xs)) * np.sum(y_preds - ys)

            self.w -= lr * dw
            self.b -= lr * db

            if epoch % 20 == 0:
                loss = np.mean((y_preds - ys) ** 2)
                print(f"epoch {epoch:3d} | loss: {loss:.6f} | w: {self.w:.4f} | b: {self.b:.4f}")


if __name__ == "__main__":
    # Data: f(x) = 2x + 1, ditambah noise kecil
    xs = [0.0, 1.0, 2.0, 3.0, 4.0]
    ys = [1.0, 3.0, 5.0, 7.0, 9.0]

    n = Neuron()
    print(f"sebelum training: x=5 -> {n.forward(5):.2f} (target: 11)")
    n.train(xs, ys, lr=0.01, epochs=200)
    print(f"setelah training: x=5 -> {n.forward(5):.2f} (target: 11)")
