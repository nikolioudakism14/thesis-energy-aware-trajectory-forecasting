"""
Generates all three figures used in the thesis (Figures 5.1, 5.2, 5.3) from
the two result CSVs produced by run_all_three_models.py:
  - thesis_three_model_results.csv
  - thesis_training_curves.csv

Usage:
    python make_figures.py
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

results = pd.read_csv("thesis_three_model_results.csv")
curves = pd.read_csv("thesis_training_curves.csv")

MODEL_ORDER = ["SOFTS", "LSTM", "Informer"]
LABELS = {"SOFTS": "SOFTS (Lightweight)", "LSTM": "LSTM (Medium)", "Informer": "Informer (Heavy)"}
COLORS = {"SOFTS": "#2E8B57", "LSTM": "#2B5B8C", "Informer": "#B9752C"}

# ------------------------------------------------------------------
# Figure 5.1: Training loss convergence
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5))
markers = {"SOFTS": "o", "LSTM": "s", "Informer": "^"}
for model in MODEL_ORDER:
    sub = curves[curves["Model"] == model].sort_values("Epoch")
    ax.plot(sub["Epoch"], sub["Train_MSE"], marker=markers[model],
            label=LABELS[model], color=COLORS[model], linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training MSE")
ax.set_title("Figure 5.1: Training Loss Convergence Across the Three Architectures")
ax.set_xticks(sorted(curves["Epoch"].unique()))
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figure_5_1_training_curves.png", dpi=220, facecolor="white")
plt.close()

# ------------------------------------------------------------------
# Figure 5.2: Energy disaggregation by hardware component
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(MODEL_ORDER))
width = 0.55

gpu_wh, ram_wh, cpu_wh, totals = [], [], [], []
for model in MODEL_ORDER:
    row = results[results["Model"] == model].iloc[0]
    total_wh = row["Energy_kWh"] * 1000
    gpu_wh.append(total_wh * row["GPU_Energy_pct"] / 100)
    ram_wh.append(total_wh * row["RAM_Energy_pct"] / 100)
    cpu_wh.append(total_wh * row["CPU_Energy_pct"] / 100)
    totals.append(total_wh)

b1 = ax.bar(x, gpu_wh, width, label="GPU (Tesla T4)", color="#2B5B8C")
b2 = ax.bar(x, ram_wh, width, bottom=gpu_wh, label="RAM", color="#5A9BD4")
bottom2 = [g + r for g, r in zip(gpu_wh, ram_wh)]
b3 = ax.bar(x, cpu_wh, width, bottom=bottom2, label="CPU (Xeon 2.0GHz)", color="#B9752C")

for i, model in enumerate(MODEL_ORDER):
    row = results[results["Model"] == model].iloc[0]
    ax.text(x[i], gpu_wh[i]/2, f"{row['GPU_Energy_pct']}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(x[i], gpu_wh[i]+ram_wh[i]/2, f"{row['RAM_Energy_pct']}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(x[i], bottom2[i]+cpu_wh[i]/2, f"{row['CPU_Energy_pct']}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(x[i], totals[i]+0.03, f"{totals[i]:.4f} Wh", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels([LABELS[m] for m in MODEL_ORDER])
ax.set_ylabel("Energy Consumed (Wh)")
ax.set_title("Figure 5.2: Energy Consumption Disaggregation by Hardware Component\n(All Three Models — 5 Epochs)")
ax.legend(loc="upper left")
ax.set_ylim(0, max(totals) * 1.22)
plt.tight_layout()
plt.savefig("figure_5_2_energy_disaggregation.png", dpi=220, facecolor="white")
plt.close()

# ------------------------------------------------------------------
# Figure 5.3: Accuracy-energy Pareto front
# ------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 5))
for model in MODEL_ORDER:
    row = results[results["Model"] == model].iloc[0]
    ax.scatter(row["Energy_kWh"], row["Val_MSE"], s=110, c=COLORS[model], zorder=3)
    ax.annotate(model, (row["Energy_kWh"], row["Val_MSE"]),
                textcoords="offset points", xytext=(8, 6), fontsize=11)

softs_row = results[results["Model"] == "SOFTS"].iloc[0]
lstm_row = results[results["Model"] == "LSTM"].iloc[0]
ax.plot([softs_row["Energy_kWh"], lstm_row["Energy_kWh"]],
        [softs_row["Val_MSE"], lstm_row["Val_MSE"]],
        "--", color="#888888", zorder=1, label="Pareto front (SOFTS \u2013 LSTM)")

ax.set_xlabel("Energy Consumption (kWh)")
ax.set_ylabel("Validation MSE")
ax.set_title("Figure 5.3: Accuracy\u2013Energy Pareto Front")
ax.legend()
plt.tight_layout()
plt.savefig("figure_5_3_pareto_front.png", dpi=220, facecolor="white")
plt.close()

print("Saved: figure_5_1_training_curves.png, figure_5_2_energy_disaggregation.png, figure_5_3_pareto_front.png")
