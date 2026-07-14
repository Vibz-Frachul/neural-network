# RNN / LSTM untuk Klasifikasi Teks dengan PyTorch
# Dataset: AG_NEWS — 4 kategori berita: World, Sports, Business, Sci/Tec.
# LSTM dipake karena teks itu sequential — urutan kata penting buat nentuin arti.

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from collections import Counter
import re

# LSTM cell punya 3 gate (input, forget, output) yang ngatur aliran informasi.
# Forget gate nentuin apa yang mau dilupain, input gate nentuin info baru apa yang disimpen,
# output gate nentuin apa yang dikeluarin. Mekanisme ini ngatasin vanishing gradient.

class LSTMText(nn.Module):
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128, num_classes=4):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=True)
        self.classifier = nn.Linear(hidden_dim * 2, num_classes)

    def forward(self, x):
        x = self.embedding(x)
        _, (hidden, _) = self.lstm(x)
        # concat hidden states dari bidirectional LSTM
        x = torch.cat((hidden[-2], hidden[-1]), dim=1)
        return self.classifier(x)


def tokenize(text, vocab=None, max_len=64):
    """Tokenisasi sederhana: lowercase, pisah kata, padding/truncation."""
    tokens = re.findall(r'\w+', text.lower())
    if vocab is not None:
        ids = [vocab.get(t, 1) for t in tokens[:max_len]]  # 1 = <UNK>
        ids += [0] * (max_len - len(ids))  # 0 = <PAD>
        return ids
    return tokens


if __name__ == "__main__":
    from sklearn.datasets import fetch_20newsgroups

    # ambil subset 4 kategori dari 20 Newsgroups
    categories = ['rec.sport.baseball', 'sci.space', 'comp.graphics', 'talk.politics.guns']
    news = fetch_20newsgroups(subset='all', categories=categories, remove=('headers', 'footers', 'quotes'))

    # build vocabulary
    all_tokens = [t for text in news.data for t in tokenize(text)]
    vocab = {word: idx+2 for idx, (word, _) in enumerate(Counter(all_tokens).most_common(5000))}
    labels = {name: i for i, name in enumerate(categories)}
    label_ids = [labels[news.target_names[t]] for t in news.target]

    # dataset
    X = torch.tensor([tokenize(t, vocab) for t in news.data], dtype=torch.long)
    y = torch.tensor(label_ids, dtype=torch.long)
    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=64, shuffle=True)

    model = LSTMText(len(vocab) + 2)
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    print("training LSTM on 4-class text classification...")
    for epoch in range(1, 6):
        model.train()
        total_loss, correct = 0.0, 0
        for texts, labels in loader:
            optimizer.zero_grad()
            loss = nn.CrossEntropyLoss()(model(texts), labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            correct += (model(texts).argmax(dim=1) == labels).sum().item()
        acc = 100.0 * correct / len(dataset)
        print(f"epoch {epoch} | loss: {total_loss/len(loader):.4f} | acc: {acc:.2f}%")
