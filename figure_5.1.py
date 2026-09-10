# ---- New figure: training loss curves, all 3 models ----
epochs = [1,2,3,4,5]
softs = [0.079713, 0.073955, 0.072891, 0.072407, 0.072581]
lstm  = [0.081007, 0.072019, 0.071382, 0.071417, 0.070947]
informer = [0.079738, 0.074624, 0.073587, 0.073121, 0.072719]

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(epochs, softs, marker="o", label="SOFTS (Lightweight)", color="#2E8B57", linewidth=2)
ax.plot(epochs, lstm, marker="s", label="LSTM (Medium)", color="#2B5B8C", linewidth=2)
ax.plot(epochs, informer, marker="^", label="Informer (Heavy)", color="#B9752C", linewidth=2)
ax.set_xlabel("Epoch")
ax.set_ylabel("Training MSE")
ax.set_title("Figure 5.1: Training Loss Convergence Across the Three Architectures")
ax.set_xticks(epochs)
ax.legend()
ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig("figure_5_1_training_curves.png", dpi=220, facecolor="white")
plt.close()
print("done")