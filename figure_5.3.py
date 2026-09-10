import matplotlib.pyplot as plt

models = ["SOFTS", "LSTM", "Informer"]
mse = [0.069691, 0.068323, 0.070225]
energy = [0.000909, 0.000968, 0.001196]
colors = ["#2E8B57", "#2B5B8C", "#B9752C"]

fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(energy, mse, s=110, c=colors, zorder=3)
for m, e, s in zip(models, energy, mse):
    ax.annotate(m, (e, s), textcoords="offset points", xytext=(8, 6), fontsize=11)

# Pareto front line: SOFTS -> LSTM (Informer is dominated, shown off the front)
ax.plot([0.000909, 0.000968], [0.069691, 0.068323], "--", color="#888888", zorder=1,
        label="Pareto front (SOFTS \u2013 LSTM)")

ax.set_xlabel("Energy Consumption (kWh)")
ax.set_ylabel("Validation MSE")
ax.set_title("Figure 5.3: Accuracy\u2013Energy Pareto Front")
ax.legend()
plt.tight_layout()
plt.savefig("figure_5_3_pareto_front.png", dpi=220)