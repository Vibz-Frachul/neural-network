# Implementasi Multi-Layer Perceptron from Scratch
# Sebuah neural network dengan 2 hidden layer, backpropagation manual pake numpy.

import numpy as np

# Arsitektur: input -> hidden(16, ReLU) -> hidden(8, ReLU) -> output(1, Sigmoid)
# Fungsi aktivasi sigmoid dipake di output karena kita klasifikasi biner.
# Hidden layer pake ReLU biar ngga vanishing gradient.

def relu(z):
    return np.maximum(0, z)

def relu_deriv(z):
    return (z > 0).astype(float)

def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

def sigmoid_deriv(z):
    s = sigmoid(z)
    return s * (1 - s)


class MLP:
    def __init__(self, input_dim=2, hidden1=16, hidden2=8, output_dim=1):
        """Init weight pake metode He — recommended buat ReLU (He et al., 2015)."""
        self.w1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros((1, hidden1))
        self.w2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros((1, hidden2))
        self.w3 = np.random.randn(hidden2, output_dim) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros((1, output_dim))

    def forward(self, x):
        """
        Forward pass: x -> z1 -> a1(ReLU) -> z2 -> a2(ReLU) -> z3 -> a3(Sigmoid).
        Simpan intermediate values buat backprop nanti.
        """
        self.z1 = x @ self.w1 + self.b1
        self.a1 = relu(self.z1)
        self.z2 = self.a1 @ self.w2 + self.b2
        self.a2 = relu(self.z2)
        self.z3 = self.a2 @ self.w3 + self.b3
        self.a3 = sigmoid(self.z3)
        return self.a3

    def backward(self, x, y, lr=0.1):
        """Backpropagation: hitung gradient tiap layer dari output ke input (chain rule)."""
        m = x.shape[0]
        y = y.reshape(-1, 1)

        # output layer (sigmoid + binary cross-entropy)
        dz3 = (self.a3 - y) / m  # ∂loss/∂z3 — turunan BCE + sigmoid udah combined
        dw3 = self.a2.T @ dz3
        db3 = np.sum(dz3, axis=0, keepdims=True)

        # hidden layer 2 (ReLU)
        dz2 = (dz3 @ self.w3.T) * relu_deriv(self.z2)
        dw2 = self.a1.T @ dz2
        db2 = np.sum(dz2, axis=0, keepdims=True)

        # hidden layer 1 (ReLU)
        dz1 = (dz2 @ self.w2.T) * relu_deriv(self.z1)
        dw1 = x.T @ dz1
        db1 = np.sum(dz1, axis=0, keepdims=True)

        # update weight
        self.w3 -= lr * dw3; self.b3 -= lr * db3
        self.w2 -= lr * dw2; self.b2 -= lr * db2
        self.w1 -= lr * dw1; self.b1 -= lr * db1


if __name__ == "__main__":
    from sklearn.datasets import make_moons

    np.random.seed(42)
    x, y = make_moons(n_samples=500, noise=0.2, random_state=42)

    model = MLP()
    for epoch in range(2000):
        pred = model.forward(x)
        loss = -np.mean(y * np.log(pred + 1e-8) + (1 - y) * np.log(1 - pred + 1e-8))
        model.backward(x, y, lr=0.5)
        if epoch % 400 == 0:
            acc = np.mean((pred > 0.5).astype(float).ravel() == y)
            print(f"epoch {epoch:4d} | loss: {loss:.4f} | acc: {acc:.4f}")

    print(f"\nfinal accuracy: {np.mean((model.forward(x) > 0.5).astype(float).ravel() == y):.4f}")
