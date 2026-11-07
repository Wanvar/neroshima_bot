
import torch
import torch.nn as nn
import torch.nn.functional as F
from game import GameState
import game
import random
random.seed(541231254) #1234

class Neuromer(nn.Module):
    def __init__(self, input_dim, num_tiles=19, d_model=128, nhead=4, num_layers=3):
        super().__init__()

        self.tile_embedding = nn.Linear(input_dim, d_model)

        # ZWIĘKSZAMY num_tiles o 1 dla pozycji, bo mamy teraz 20 obiektów!
        self.pos_embedding = nn.Embedding(num_tiles + 1, d_model)

        # NOWOŚĆ: Definiujemy token [CLS] jako parametr, który sieć będzie trenować
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))


        # Transformer pozostaje bez zmian
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.15, activation='relu', batch_first=True,
            norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.head = nn.Sequential(
            nn.Linear(d_model *2, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, num_tiles, _ = x.shape

        # 1. Standardowy embedding kafelków: [Batch, 19, d_model]
        x_emb = self.tile_embedding(x)

        # 2. Powielamy token [CLS] dla całego batcha: [Batch, 1, d_model]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)

        # 3. Sklejamy: CLS na początek! Wynik: [Batch, 20, d_model]
        x = torch.cat((cls_tokens, x_emb), dim=1)

        # 4. Dodajemy pozycję (teraz dla 20 elementów)
        pos_indices = torch.arange(num_tiles + 1, device=x.device).unsqueeze(0).expand(batch_size, -1)
        p_emb = self.pos_embedding(pos_indices)
        x = x + p_emb

        # 5. Przepuszczamy przez Transformer
        x = self.transformer(x)  # Kształt: [Batch, 20, d_model]

        x_cls = x[:, 0, :]
        x_max = x[:, 1:, :].max(dim=1)[0]
       # x_std = torch.std(x[:, 1:, :], dim=1) + 1e-6

        # Weighted Mean (z wykorzystaniem warstwy uwag)

        #x_mean = x[:, 1:, :].mean(dim=1)

        # Sklejenie 4 ekspertów
        x_combined = torch.cat((x_cls, x_max), dim=1)  # [Batch, 1024]
        #x = x.mean(dim = 1)
        # 3. Podajesz do zmodyfikowanego MLP Head:
        return self.head(x_combined)


class Neuromer2(nn.Module):
    def __init__(self, input_dim, num_tiles=19, d_model=128, nhead=4, num_layers=3, num_transitions=32):
        super().__init__()
        self.num_transitions = num_transitions

        # ==========================================
        # 1. BAZOWY ENKODER STANU (Twoja architektura)
        # ==========================================
        self.tile_embedding = nn.Linear(input_dim, d_model)
        self.pos_embedding = nn.Embedding(num_tiles + 1, d_model)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, d_model))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
            dropout=0.07, activation='relu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # Kompresja 4 ekspertów z powrotem do d_model
        self.state_compressor = nn.Linear(d_model * 4, d_model)

        # ==========================================
        # 2. WEWNĘTRZNY MINIBATCH (32 Przejścia)
        # ==========================================
        self.traj_pos_embedding = nn.Embedding(num_transitions, d_model)
        #
        # # Mniejszy transformer do przeliczania trajektorii
        # traj_layer = nn.TransformerEncoderLayer(
        #     d_model=d_model, nhead=nhead, dim_feedforward=d_model * 2,
        #     dropout=0.1, activation='relu', batch_first=True, norm_first=True
        # )
        # # 2 warstwy zazwyczaj wystarczą do modelowania ukrytych dynamik
        # self.traj_transformer = nn.TransformerEncoder(traj_layer, num_layers=2)
        self.alternative_processor = nn.Sequential(
            nn.Linear(d_model, d_model * 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model)
        )
        # ==========================================
        # 3. SPRYTNA SELEKCJA I GŁOWA WYJŚCIOWA
        # ==========================================
        # Warstwa oceniająca "ważność" każdego z 32 kroków (Attention Pooling)
        self.attention_scorer = nn.Linear(d_model, 1)

        self.head = nn.Sequential(
            nn.Linear(d_model, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)
        )

    def forward(self, x):
        batch_size, num_tiles, _ = x.shape

        # --- KROK 1: Przetwarzanie bazowe ---
        x_emb = self.tile_embedding(x)
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x_seq = torch.cat((cls_tokens, x_emb), dim=1)

        pos_indices = torch.arange(num_tiles + 1, device=x.device).unsqueeze(0).expand(batch_size, -1)
        x_seq = x_seq + self.pos_embedding(pos_indices)
        x_seq = self.transformer(x_seq)

        # Wyciągnięcie 4 ekspertów
        x_cls = x_seq[:, 0, :]
        x_max = x_seq[:, 1:, :].max(dim=1)[0]
        x_std = torch.std(x_seq[:, 1:, :], dim=1) + 1e-6
        x_mean = x_seq[:, 1:, :].mean(dim=1)

        # Sklejenie i kompresja: [Batch, 4*d_model] -> [Batch, d_model]
        x_combined = torch.cat((x_cls, x_max, x_std, x_mean), dim=1)
        base_state = self.state_compressor(x_combined)

        # --- KROK 2: Broadcast i Trajektorie ---
        # Kopiujemy stan na 32 warianty: [Batch, 32, d_model]
        state_expanded = base_state.unsqueeze(1).expand(-1, self.num_transitions, -1)

        # Dodajemy embeddingi przejść
        traj_indices = torch.arange(self.num_transitions, device=x.device).unsqueeze(0).expand(batch_size, -1)
        state_with_traj = state_expanded + self.traj_pos_embedding(traj_indices)

        # --- KROK 3: Przetwarzanie Minibatcha ---
        # Zależnie od zadania, transformer tutaj pozwoli na interakcję między tymi 32 stanami.
        # Jeśli mają być całkowicie niezależne, możesz zastąpić to zwykłym MLP nałożonym na wymiar d_model.
        #traj_processed = self.traj_transformer(state_with_traj)  # [Batch, 32, d_model]
        traj_processed = self.alternative_processor(state_with_traj)  #
        # --- KROK 4: Sprytna Selekcja (Attention Pooling) ---
        # Liczymy wagi dla każdego z 32 przejść [Batch, 32, 1]
        attn_logits = self.attention_scorer(traj_processed)
        attn_weights = torch.softmax(attn_logits, dim=1)

        # Kumulujemy wynik, mnożąc przejścia przez ich wagi i sumując
        # Zamiast hard-selection, mamy zróżniczkowalną fuzję najlepszych wariantów.
        selected_state = torch.sum(traj_processed * attn_weights, dim=1)  # [Batch, d_model]

        # Alternatywa (Max-Pooling), jeśli wolisz ostre cięcie (odkomentuj poniżej):
        # selected_state = traj_processed.max(dim=1)[0]

        # --- KROK 5: Ocena końcowa ---
        return self.head(selected_state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Using device: {device}")

if torch.cuda.is_available():
    print(torch.cuda.get_device_name(0))


model = Neuromer(
    input_dim=game.TileInfo.data_len,
    num_tiles=19,
    d_model=128,  # rozmiar ukryty wektora dla każdego kafelka
    nhead=4,      # liczba główek attention
    num_layers=4  # głębokość transformera
).to(device)


optimizer = torch.optim.AdamW( # AdamW jest zazwyczaj lepszy dla Transformerów niż zwykły Adam
    model.parameters(),
    lr=1e-3,                   # Transformery lubią odrobinę mniejszy lr na start (np. 5e-4 lub 1e-3)
    weight_decay=1e-4
)

criterion = nn.MSELoss()
@torch.no_grad()
def check_pairwise_accuracy(model, state_generator, num_samples=30_000, num_pairs=10000):
    model.eval()
    print("\n=== URUCHAMIANIE METRYKI PAIRWISE ACCURACY ===")

    # 1. Generujemy świeżą, niezależną paczkę danych testowych
    X_test, Y_test = state_generator.get_data(num_samples)

    # Przekształcamy X na tensor i przepuszczamy przez model (w oryginalnych wymiarach 3D!)
    X_tensor = torch.tensor(X_test, dtype=torch.float32).to(device)
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



from torch.utils.data import TensorDataset, DataLoader
s = GameState()


# items = s.get_data2(30)
# # 2. Wczytujesz słownik wag na odpowiednie urządzenie (np. CPU lub CUDA)
#model.load_state_dict(torch.load("neuromer_model_256_8_6.pt", map_location=device))
#pairwise_acc = check_pairwise_accuracy(model, state_generator=s, num_samples=100_000, num_pairs=100_000)
# # 3. Przełączasz model w tryb ewaluacji (wyłącza to Dropout)
# model.eval()
# with torch.no_grad():
#     for x, y, d in zip(*items):
#         #print(d[0])
#         #print(d[1])
#
#         # 1. Tworzymy tensor i od razu dodajemy wymiar batcha za pomocą .unsqueeze(0)
#         # Kształt zmienia się z [19, data_len] na [1, 19, data_len]
#         x_tensor = torch.tensor(x, dtype=torch.float32).unsqueeze(0).to(device)
#
#         # Jeśli używasz wersji z kontekstem globalnym/dodatkowym,
#         # pamiętaj, by je też zamienić na tensory z unsqueeze(0)!
#         # context_tensor = torch.tensor(d, dtype=torch.float32).unsqueeze(0).to(device)
#
#         print(f"wynik {y}")
#
#         # 2. POPRAWKA: Wywołujemy model(x_tensor), a NIE model.forward(x_tensor)
#         prediction = model(x_tensor)
#         print(f"sieć: {prediction}")
#
#     abc = input("koniec")

import time
start = time.time()
sample_size = 1_000_000

X, Y = s.get_data(sample_size)

end = time.time()

print(f"czas generowania danych: {end - start}")
print(Y[0:10])
# poprawne indeksowanie

import numpy as np

# X = loaded['x']
# Y = loaded['y']




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




# ... (Generowanie X i Y przez s.get_data(sample_size) pozostaje bez zmian) ...

split_idx = int(0.8 * sample_size)

# Numpy -> Torch twardo zachowując wymiary 3D
X_train = torch.tensor(X[:split_idx], dtype=torch.float32)
Y_train = torch.tensor(Y[:split_idx], dtype=torch.float32)

X_eval = torch.tensor(X[split_idx:], dtype=torch.float32)
Y_eval = torch.tensor(Y[split_idx:], dtype=torch.float32)




# Standardowy dataset i loader (bardzo szybki)
train_dataset = TensorDataset(X_train, Y_train)
eval_dataset = TensorDataset(X_eval, Y_eval)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
eval_loader = DataLoader(eval_dataset, batch_size=64, shuffle=False)

import torch
import time
import numpy as np
import pandas as pd

batch_sizes = [1, 4, 16, 64, 256, 1024]

results = []

model.eval()

with torch.no_grad():

    # warmup
    x = X_train[:128].to(device)
    for _ in range(100):
        model(x)

    torch.cuda.synchronize()

    for bs in batch_sizes:

        x = X_train[:bs].to(device)

        # dodatkowy warmup dla danego rozmiaru
        for _ in range(20):
            model(x)

        torch.cuda.synchronize()

        times = []

        for _ in range(100):

            torch.cuda.synchronize()
            start = time.perf_counter()

            model(x)

            torch.cuda.synchronize()
            end = time.perf_counter()

            times.append((end - start) * 1000)

        results.append({
            "batch": bs,
            "mean_ms": np.mean(times),
            "median_ms": np.median(times),
            "p95_ms": np.percentile(times, 95),
            "throughput_samples_s": bs / (np.mean(times) / 1000)
        })

df = pd.DataFrame(results)

print(df.to_string(index=False))
def train_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device) # Kształt: [128, 19, data_len]
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = criterion(pred, y_batch)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(loader)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    total_loss = 0.0
    debug_pred, debug_target = None, None

    for X_batch, y_batch in loader:
        X_batch = X_batch.to(device)
        y_batch = y_batch.to(device)

        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        total_loss += loss.item()

        if debug_pred is None:
            debug_pred = pred[0].item()
            debug_target = y_batch[0].item()

    return total_loss / len(loader), debug_pred, debug_target

epochs = 30

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





pairwise_acc = check_pairwise_accuracy(model, state_generator=s, num_samples=30_000, num_pairs=10_000)

# Definiujemy ścieżkę zapisu (standardowe rozszerzenie to .pt lub .pth)
PATH = "null"

# Zapisujemy SAMO state_dict
torch.save(model.state_dict(), PATH)
print("Model zapisany pomyślnie!")
