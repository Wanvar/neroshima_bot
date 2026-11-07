import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader # Zastępuje DataLoader z torch.utils.data


import game
from game import GameState

class ValueGNN(nn.Module): #<--- ten model chcę wprowadzić! batch chyba niepotrzebnym, graf jest spójny, niekurunkowy, w pełni połączony
    def __init__(self, input_dim):
        super().__init__()

        self.conv1 = GCNConv(input_dim, 128)
        self.conv2 = GCNConv(128, 128)
        self.conv3 = GCNConv(128, 64)

        self.head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, 1),
            #nn.Tanh()
        )

    def forward(self, x, edge_index, batch):
        x = self.conv1(x, edge_index)
        x = F.relu(x)

        x = self.conv2(x, edge_index)
        x = F.relu(x)

        x = self.conv3(x, edge_index)
        x = F.relu(x)

        # pooling całego grafu → jedna wartość
        x = global_mean_pool(x, batch)

        return self.head(x)


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

# model = ValueMLP(
#     input_dim=19 * game.TileInfo.data_len
# ).to(device)
model = ValueGNN(
     input_dim=game.TileInfo.data_len
).to(device)


criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=1e-4,
    #weight_decay=1e-4
)


# =========================
# DATA
# =========================

s = GameState()
edges = s.get_edges()

import time
start = time.time()
sample_size = 100_000

X, Y = s.get_data(sample_size)

end = time.time()
print(Y[0:10])
print(f"czas generowania danych: {end - start}")
# poprawne indeksowanie
edge_index = torch.tensor(list(edges), dtype=torch.long).T
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

data_list = []
for i in range(sample_size):
    # X[i] ma już idealny kształt (19, data_len) - wystarczy zrzutować na tensor
    x_tensor = torch.tensor(X[i], dtype=torch.float32)
    y_tensor = torch.tensor(Y[i], dtype=torch.float32)  # Y[i] to tablica 1-elementowa

    data = Data(x=x_tensor, edge_index=edge_index, y=y_tensor)
    data_list.append(data)

# Podział (zakładając split_idx = int(0.8 * sample_size))
split_idx = int(0.8 * sample_size)

train_dataset = data_list[:split_idx]
eval_dataset = data_list[split_idx:]

# Tworzymy Loadery z PyG
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

    for batch_data in loader:
        # Przenosimy cały obiekt batch_data na GPU
        batch_data = batch_data.to(device)

        # PyG sprytnie łączy grafy, podając wszystko w 3 tensorach:
        pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)

        # Upewniamy się, że wymiary predykcji i celu do siebie pasują
        loss = criterion(pred.view(-1), batch_data.y.view(-1))

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

    for batch_data in loader:
        batch_data = batch_data.to(device)

        pred = model(batch_data.x, batch_data.edge_index, batch_data.batch)
        loss = criterion(pred.view(-1), batch_data.y.view(-1))

        total_loss += loss.item()

        # debug sample
        if debug_pred is None:
            debug_pred = pred[0].item()
            debug_target = batch_data.y[0].item()

    return (
        total_loss / len(loader),
        debug_pred,
        debug_target
    )

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
def check_pairwise_accuracy_gnn(model, state_generator, edge_index, num_samples=30_000, num_pairs=10000):
    model.eval()
    print("\n=== URUCHAMIANIE METRYKI PAIRWISE ACCURACY (GNN) ===")

    # 1. Generujemy świeżą paczkę danych testowych
    X_test, Y_test = state_generator.get_data(num_samples)  # X_test ma kształt: [30000, 19, Cechy]
    targets = Y_test.flatten()

    # 2. BUDOWANIE GRAFÓW I PARAMETRU BATCH
    # Tworzymy listę pojedynczych obiektów grafowych PyG
    graph_list = []
    for i in range(num_samples):
        # Pobieramy cechy kafelków dla i-tej planszy [19, Cechy]
        x_features = torch.tensor(X_test[i], dtype=torch.float32)

        # Tworzymy pojedynczy graf
        graph = Data(x=x_features, edge_index=edge_index)
        graph_list.append(graph)

    # Magia PyG: Batch.from_data_list automatycznie sklei grafy
    # i wygeneruje tensor 'batch', o który pytałeś!
    pyg_batch = Batch.from_data_list(graph_list).to(device)

    # 3. Przepuszczamy przez model GNN
    # Twój model GNN w forward() powinien teraz przyjmować: model(batch.x, batch.edge_index, batch.batch)
    predictions = model(pyg_batch.x, pyg_batch.edge_index, pyg_batch.batch).cpu().numpy().flatten()

    # 4. Szukamy par o różnych wynikach (y1 != y2) - reszta kodu bez zmian
    correct_predictions = 0
    valid_pairs_found = 0
    attempts = 0
    max_attempts = num_pairs * 10

    while valid_pairs_found < num_pairs and attempts < max_attempts:
        attempts += 1
        idx1 = random.randint(0, num_samples - 1)
        idx2 = random.randint(0, num_samples - 1)

        y1, y2 = targets[idx1], targets[idx2]

        if y1 != y2:
            pred1, pred2 = predictions[idx1], predictions[idx2]
            if (y1 > y2) == (pred1 > pred2):
                correct_predictions += 1
            valid_pairs_found += 1

    if valid_pairs_found == 0:
        print("Błąd: Nie znaleziono żadnych par o różnych wynikach Y!")
        return 0.0

    accuracy = (correct_predictions / valid_pairs_found) * 100
    print(f"Przeanalizowano unikalnych par: {valid_pairs_found}")
    print(f"Ostateczny wynik -> Pairwise Accuracy: {accuracy:.2f}%")

    return accuracy


# WYWOŁANIE:
# Musisz podać 'edge_index' Twojej planszy Neuroshimy (np. wygenerowany z sąsiedztwa hexów)
pairwise_acc = check_pairwise_accuracy_gnn(
    model,
    state_generator=s,
    edge_index=edge_index,  # Twój tensor połączeń planszy
    num_samples=30_000,
    num_pairs=10_000
)