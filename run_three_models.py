import os, time, math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from codecarbon import EmissionsTracker

# ------------------------------------------------------------------
# 0. CONFIG
# ------------------------------------------------------------------
DATA_PATH = "/content/londonlandingsync_all.csv"
SEQ_LEN = 10
EPOCHS = 5
BATCH = 64
LR = 1e-3
SEED = 42
OUT_CSV = "thesis_three_model_results.csv"

torch.manual_seed(SEED)
np.random.seed(SEED)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Real dataset not found at {DATA_PATH}. Point DATA_PATH at the "
        f"actual London Landing Sync CSV — this script will not fall back "
        f"to synthetic data."
    )


# ------------------------------------------------------------------
# 1. REAL-DATA FEATURE ENGINEERING
# ------------------------------------------------------------------
def load_real_telemetry(csv_path):
    col_names = ['flight_id', 'seq_id', 'timestamp', 'lon', 'lat', 'alt']
    df = pd.read_csv(csv_path, header=None, names=col_names)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values(by=['flight_id', 'timestamp']).reset_index(drop=True)

    dt = df.groupby('flight_id')['timestamp'].diff().dt.total_seconds().fillna(1.0)
    dt = np.where(dt <= 0, 1.0, dt)

    dlat = df.groupby('flight_id')['lat'].diff().fillna(0) * 111000
    dlon = df.groupby('flight_id')['lon'].diff().fillna(0) * 111000 * np.cos(np.radians(df['lat']))
    dalt = df.groupby('flight_id')['alt'].diff().fillna(0)

    df['speed'] = np.sqrt(dlat ** 2 + dlon ** 2) / dt
    df['heading'] = np.degrees(np.arctan2(dlon, dlat)) % 360
    df['vert_rate'] = dalt / dt
    return df


class TelemetryDataset(Dataset):
    def __init__(self, df, seq_len=SEQ_LEN, pred_len=1):
        features = ['lat', 'lon', 'alt', 'speed', 'heading', 'vert_rate']
        data = df[features].fillna(0).values.astype(np.float32)
        mean, std = data.mean(axis=0), data.std(axis=0) + 1e-8
        data = (data - mean) / std

        x, y = [], []
        for i in range(len(data) - seq_len - pred_len + 1):
            x.append(data[i:i + seq_len])
            y.append(data[i + seq_len:i + seq_len + pred_len, :2])

        self.x = torch.tensor(np.array(x), dtype=torch.float32)
        self.y = torch.tensor(np.array(y), dtype=torch.float32).squeeze(1)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]


# ------------------------------------------------------------------
# 2. MODELS — SOFTS, LSTM, Informer
# ------------------------------------------------------------------
class LSTMBaseline(nn.Module):
    """Matches the architecture already narrated in Chapter 5.1: 2-layer
    LSTM (hidden=64), single linear projection head."""

    def __init__(self, input_dim=6, hidden_dim=64, num_layers=2, output_dim=2):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.fc(out[:, -1, :])


class SOFTSModule(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, input_dim=6, d_model=64, d_core=32, output_dim=2):
        super().__init__()
        self.seq_len = seq_len
        self.input_projection = nn.Linear(input_dim, d_model)
        self.core_mlp = nn.Sequential(nn.Linear(d_model, d_core), nn.GELU(), nn.Linear(d_core, d_model))
        self.fusion = nn.Linear(d_model * 2, d_model)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * d_model, 128), nn.GELU(),
                                  nn.Linear(128, output_dim))

    def forward(self, x):
        h = self.input_projection(x)
        core = torch.mean(h, dim=1, keepdim=True)
        core = self.core_mlp(core).expand(-1, self.seq_len, -1)
        fused = torch.cat([h, core], dim=-1)
        out_h = self.norm(h + self.fusion(fused))
        return self.head(out_h)


class ProbSparseAttention(nn.Module):
    def __init__(self, d_model=64, n_heads=4):
        super().__init__()
        self.n_heads, self.d_k = n_heads, d_model // n_heads
        self.q_dense = nn.Linear(d_model, d_model)
        self.k_dense = nn.Linear(d_model, d_model)
        self.v_dense = nn.Linear(d_model, d_model)
        self.out_dense = nn.Linear(d_model, d_model)

    def forward(self, x):
        b, l, d = x.shape
        H, d_k = self.n_heads, self.d_k
        q = self.q_dense(x).view(b, l, H, d_k).transpose(1, 2)
        k = self.k_dense(x).view(b, l, H, d_k).transpose(1, 2)
        v = self.v_dense(x).view(b, l, H, d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(d_k)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, v).transpose(1, 2).contiguous().view(b, l, d)
        return self.out_dense(out)


class InformerBaseline(nn.Module):
    def __init__(self, seq_len=SEQ_LEN, input_dim=6, d_model=64, n_heads=4, output_dim=2):
        super().__init__()
        self.embedding = nn.Linear(input_dim, d_model)
        self.attn = ProbSparseAttention(d_model, n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(seq_len * d_model, 64), nn.ReLU(), nn.Linear(64, output_dim))

    def forward(self, x):
        h = self.embedding(x)
        h = self.norm(h + self.attn(h))
        return self.head(h)


# ------------------------------------------------------------------
# 3. TRAIN / EVAL / ENERGY-TRACK ONE MODEL
# ------------------------------------------------------------------
def evaluate(model, loader, device):
    model.eval()
    mse_sum = mae_sum = n = 0
    with torch.no_grad():
        for bx, by in loader:
            bx, by = bx.to(device), by.to(device)
            pred = model(bx)
            mse_sum += F.mse_loss(pred, by).item() * bx.size(0)
            mae_sum += F.l1_loss(pred, by).item() * bx.size(0)
            n += bx.size(0)
    return mse_sum / n, mae_sum / n


def run_one_model(tier, model_name, model, train_loader, val_loader, device):
    print(f"\n=== {tier} tier: {model_name} ===")
    model = model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=LR)

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()

    os.makedirs("./codecarbon_logs", exist_ok=True)
    tracker = EmissionsTracker(project_name=model_name, output_dir="./codecarbon_logs",
                               save_to_file=True, log_level="error")
    tracker.start()
    t0 = time.time()

    for epoch in range(1, EPOCHS + 1):
        model.train()
        running = 0.0
        for bx, by in train_loader:
            bx, by = bx.to(device), by.to(device)
            opt.zero_grad()
            loss = F.mse_loss(model(bx), by)
            loss.backward()
            opt.step()
            running += loss.item() * bx.size(0)
        print(f"  Epoch {epoch}/{EPOCHS}  Train MSE: {running / len(train_loader.dataset):.6f}")

    train_time = time.time() - t0
    emissions_kg = tracker.stop()
    ed = tracker.final_emissions_data

    val_mse, val_mae = evaluate(model, val_loader, device)
    peak_vram_mb = torch.cuda.max_memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0

    energy_kwh = getattr(ed, "energy_consumed", 0.0) or 0.0
    gpu_kwh = getattr(ed, "gpu_energy", 0.0) or 0.0
    ram_kwh = getattr(ed, "ram_energy", 0.0) or 0.0
    cpu_kwh = getattr(ed, "cpu_energy", 0.0) or 0.0
    total_component = (gpu_kwh + ram_kwh + cpu_kwh) or 1e-12

    eei = 1.0 / (val_mse * energy_kwh) if val_mse * energy_kwh > 0 else 0.0

    row = {
        "Tier": tier, "Model": model_name, "Val_MSE": round(val_mse, 6),
        "Val_MAE": round(val_mae, 6), "Train_Time_Sec": round(train_time, 2),
        "Peak_VRAM_MB": round(peak_vram_mb, 2), "Energy_kWh": round(energy_kwh, 8),
        "GPU_Energy_pct": round(100 * gpu_kwh / total_component, 1),
        "RAM_Energy_pct": round(100 * ram_kwh / total_component, 1),
        "CPU_Energy_pct": round(100 * cpu_kwh / total_component, 1),
        "CO2_g": round(emissions_kg * 1000, 4), "EEI_MSEinv_per_kWh": round(eei, 2)
    }
    print(f"  -> Val MSE={row['Val_MSE']}  Val MAE={row['Val_MAE']}  "
          f"Energy={row['Energy_kWh']} kWh  CO2={row['CO2_g']} g  EEI={row['EEI_MSEinv_per_kWh']}")
    return row


# ------------------------------------------------------------------
# 4. MAIN — trains and prints ALL THREE models, nothing hardcoded
# ------------------------------------------------------------------
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    df = load_real_telemetry(DATA_PATH)
    dataset = TelemetryDataset(df)
    n_train = int(0.8 * len(dataset))
    n_val = len(dataset) - n_train
    gen = torch.Generator().manual_seed(SEED)
    train_ds, val_ds = torch.utils.data.random_split(dataset, [n_train, n_val], generator=gen)

    train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=BATCH, shuffle=False)

    models = [
        ("Lightweight", "SOFTS", SOFTSModule()),
        ("Medium", "LSTM", LSTMBaseline()),
        ("Heavy", "Informer", InformerBaseline()),
    ]

    results = [run_one_model(t, n, m, train_loader, val_loader, device) for t, n, m in models]

    df_out = pd.DataFrame(results)
    df_out.to_csv(OUT_CSV, index=False)

    print("\n" + "=" * 70)
    print("FINAL RESULTS — ALL THREE MODELS")
    print("=" * 70)
    print(df_out.to_string(index=False))
    print(f"\nSaved -> {OUT_CSV}")