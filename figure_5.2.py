import matplotlib.pyplot as plt
import numpy as np

# ---- Figure 5.1: stacked energy disaggregation bar chart, all 3 models ----
models = ["SOFTS\n(Lightweight)", "LSTM\n(Medium)", "Informer\n(Heavy)"]
gpu_wh = [0.6363, 0.6883, 0.8540]
ram_wh = [0.2145, 0.2178, 0.2667]
cpu_wh = [0.0582, 0.0620, 0.0753]
totals = [g+r+c for g,r,c in zip(gpu_wh, ram_wh, cpu_wh)]
gpu_pct = [70.0, 71.1, 71.4]
ram_pct = [23.6, 22.5, 22.3]
cpu_pct = [6.4, 6.4, 6.3]

fig, ax = plt.subplots(figsize=(8, 5.5))
x = np.arange(len(models))
width = 0.55

b1 = ax.bar(x, gpu_wh, width, label="GPU (Tesla T4)", color="#2B5B8C")
b2 = ax.bar(x, ram_wh, width, bottom=gpu_wh, label="RAM", color="#5A9BD4")
bottom2 = [g+r for g,r in zip(gpu_wh, ram_wh)]
b3 = ax.bar(x, cpu_wh, width, bottom=bottom2, label="CPU (Xeon 2.0GHz)", color="#B9752C")

# annotate segment percentages
for i in range(3):
    ax.text(x[i], gpu_wh[i]/2, f"{gpu_pct[i]}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(x[i], gpu_wh[i]+ram_wh[i]/2, f"{ram_pct[i]}%", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    ax.text(x[i], bottom2[i]+cpu_wh[i]/2, f"{cpu_pct[i]}%", ha="center", va="center", color="white", fontsize=8, fontweight="bold")
    ax.text(x[i], totals[i]+0.03, f"{totals[i]:.4f} Wh", ha="center", va="bottom", fontsize=10, fontweight="bold")

ax.set_xticks(x)
ax.set_xticklabels(models)
ax.set_ylabel("Energy Consumed (Wh)")
ax.set_title("Figure 5.2: Energy Consumption Disaggregation by Hardware Component\n(All Three Models — 5 Epochs)")
ax.legend(loc="upper left")
ax.set_ylim(0, max(totals)*1.22)
plt.tight_layout()
plt.savefig("figure_5_2_energy_disaggregation.png", dpi=220, facecolor="white")
plt.close()