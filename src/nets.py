import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool


import game
from game import GameState


class ValueMLP(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(256, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(128, 64),
            nn.ReLU(),

            nn.Linear(64, 1),
            #nn.Tanh()
        )

        # inicjalizacja (stabilniejszy start)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.net(x)


# =========================
# DEVICE
# =========================

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


# =========================
# MODEL
# =========================

model = ValueMLP(
    input_dim=19 * game.TileInfo.data_len
).to(device)



criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=5e-4,
    weight_decay=1e-4
)


# =========================
# DATA
# =========================

s = GameState()
edges = s.get_edges()
edge_index = torch.tensor(list(edges)).T
import time
start = time.time()
sample_size = 100_000

X, Y = s.get_data(sample_size)
print(Y[0:10])
end = time.time()

print(f"czas generowania danych: {end - start}")
# poprawne indeksowanie
split_idx = int(0.8 * sample_size)

import numpy as np



# 2. Liczymy standardowe statystyki
Y_mean = np.mean(Y)
Y_std = np.std(Y)

print(f"Przed skalowaniem -> Średnia: {Y_mean:.4f}, STD: {Y_std:.4f}")

# 3. Klasyczna standaryzacja (Z-score)
# Nie używamy np.clip! Dane mogą naturalnie przyjąć wartości np. -2.5, 3.1 itd.
Y_scaled = (Y - Y_mean) / Y_std

print(f"Po skalowaniu -> Min: {Y_scaled.min():.4f}, Max: {Y_scaled.max():.4f}")
print(f"Po skalowaniu -> Nowa Średnia: {np.mean(Y_scaled):.4f}, Nowe STD: {np.std(Y_scaled):.4f}")

Y = Y_scaled

X_train, X_eval = X[:split_idx], X[split_idx:]
Y_train, Y_eval = Y[:split_idx], Y[split_idx:]


# numpy -> torch

X_train = torch.tensor(X_train, dtype=torch.float32)
Y_train = torch.tensor(Y_train, dtype=torch.float32)

X_eval = torch.tensor(X_eval, dtype=torch.float32)
Y_eval = torch.tensor(Y_eval, dtype=torch.float32)


from torch.utils.data import TensorDataset, DataLoader

# =========================
# DATASET
# =========================

train_dataset = TensorDataset(X_train, Y_train)
eval_dataset = TensorDataset(X_eval, Y_eval)

train_loader = DataLoader(
    train_dataset,
    batch_size=128,
    shuffle=True
)

eval_loader = DataLoader(
    eval_dataset,
    batch_size=128,
    shuffle=False
)


# =========================
# TRAIN EPOCH
# =========================

def train_epoch(model, loader):
    model.train()

    total_loss = 0.0

    for X_batch, y_batch in loader:

        X_batch = X_batch.reshape(X_batch.shape[0], -1).to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)

        loss = criterion(pred, y_batch)

        optimizer.zero_grad()

        loss.backward()

        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


# =========================
# EVAL
# =========================

@torch.no_grad()
def evaluate(model, loader):
    model.eval()

    total_loss = 0.0

    debug_pred = None
    debug_target = None

    for X_batch, y_batch in loader:

        X_batch = X_batch.reshape(X_batch.shape[0], -1).to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)

        loss = criterion(pred, y_batch)

        total_loss += loss.item()

        # debug sample
        if debug_pred is None:
            debug_pred = pred[0].item()
            debug_target = y_batch[0].item()

    return (
        total_loss / len(loader),
        debug_pred,
        debug_target
    )


# =========================
# TRAIN LOOP
# =========================

epochs = 100

for epoch in range(epochs):

    train_loss = train_epoch(
        model,
        train_loader
    )

    eval_loss, pred_value, target_value = evaluate(
        model,
        eval_loader
    )

    print(
        f"[Epoch {epoch + 1:03d}] "
        f"train_loss={train_loss:.6f} "
        f"eval_loss={eval_loss:.6f} "
        f"target={target_value:.3f} "
        f"pred={pred_value:.3f}"
    )



import random


@torch.no_grad()
def check_pairwise_accuracy(model, state_generator, num_samples=30_000, num_pairs=10000):
    model.eval()
    print("\n=== URUCHAMIANIE METRYKI PAIRWISE ACCURACY ===")

    # 1. Generujemy świeżą, niezależną paczkę danych testowych
    X_test, Y_test = state_generator.get_data(num_samples)

    # Przekształcamy X na tensor i przepuszczamy przez model (w oryginalnych wymiarach 3D!)
    X_tensor = torch.tensor(X_test.reshape(X_test.shape[0], -1), dtype=torch.float32).to(device)
    predictions = model(X_tensor).cpu().numpy().flatten()  # Predykcje modelu
    targets = Y_test.flatten()  # Prawdziwe wyniki bitew

    # 2. Szukamy par o różnych wynikach (y1 != y2)
    correct_predictions = 0
    valid_pairs_found = 0

    # Aby uniknąć nieskończonej pętli, jeśli par byłoby mało, zabezpieczamy się licznikiem prób
    attempts = 0
    max_attempts = num_pairs * 10

    while valid_pairs_found < num_pairs and attempts < max_attempts:
        attempts += 1

        # Losujemy dwa różne indeksy z wygenerowanej paczki
        idx1 = random.randint(0, num_samples - 1)
        idx2 = random.randint(0, num_samples - 1)

        y1, y2 = targets[idx1], targets[idx2]

        # Sprawdzamy warunek: y1 != y2
        if y1 != y2:
            pred1, pred2 = predictions[idx1], predictions[idx2]

            # Sprawdzamy, czy relacja większości się zgadza
            # (y1 > y2) musi dać taki sam wynik logiczny jak (pred1 > pred2)
            if (y1 > y2) == (pred1 > pred2):
                correct_predictions += 1

            valid_pairs_found += 1

    # 3. Wyświetlamy wynik
    if valid_pairs_found == 0:
        print("Błąd: Nie znaleziono żadnych par o różnych wynikach Y!")
        return 0.0

    accuracy = (correct_predictions / valid_pairs_found) * 100
    print(f"Przeanalizowano unikalnych par: {valid_pairs_found}")
    print(f"Ostateczny wynik -> Pairwise Accuracy: {accuracy:.2f}%")

    return accuracy
pairwise_acc = check_pairwise_accuracy(model, state_generator=s, num_samples=30_000, num_pairs=10_000)