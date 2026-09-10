# Energy-Aware Benchmarking of Trajectory Forecasting Architectures

Code accompanying the MSc thesis *"Evaluation of the energy consumption aspects of trajectory-related AI algorithms"*. 
Benchmarks three architectures — **SOFTS** (Lightweight), **LSTM** (Medium), and **Informer**
(Heavy) — on the London Landing Sync ADS-B trajectory dataset, measuring both
predictive accuracy (MSE/MAE) and energy consumption / carbon footprint (via
[CodeCarbon](https://github.com/mlco2/codecarbon)) under an identical training
protocol, and identifying the empirical accuracy–energy Pareto front between
them.

## Results

All three models are trained under an identical protocol (batch size 64,
Adam, η=10⁻³, 5 epochs, sequence length L=10) on an NVIDIA Tesla T4 GPU.

| Model Tier | Architecture | Val. MSE | Val. MAE | Train Time (s) | Peak VRAM (MB) | Energy (kWh) | CO₂eq (g) | EEI (MSE⁻¹/kWh) |
|---|---|---|---|---|---|---|---|---|
| Lightweight | SOFTS | 0.069691 | 0.062172 | 77.09 | 19.53 | 0.000909 | 0.3173 | 15792.82 |
| Medium | LSTM | 0.068323 | 0.050057 | 78.35 | 41.04 | 0.000968 | 0.3379 | 15126.60 |
| Heavy | Informer | 0.070225 | 0.055083 | 96.22 | 20.67 | 0.001196 | 0.4177 | 11905.72 |

**Key finding**: Informer is strictly Pareto-dominated by both SOFTS and LSTM
— it has both a higher validation MSE and a higher energy cost than either
alternative, so there is no accuracy justification for its additional energy
cost at this sequence length. The genuine trade-off is between SOFTS (best
energy efficiency) and LSTM (best accuracy, particularly MAE). Full analysis
in Chapter 5 of the thesis.

## Repository structure

```
.
├── run_all_three_models.py          # trains SOFTS, LSTM, Informer end-to-end
├── make_figures.py                  # regenerates Figures 5.1, 5.2, 5.3 from the CSVs below
├── requirements.txt
├── thesis_three_model_results.csv   # summary metrics used in Table 5.1
├── thesis_training_curves.csv       # per-epoch training loss used in Figure 5.1
└── data/                            # place the London Landing Sync CSV here (not tracked)
```

`thesis_three_model_results.csv` and `thesis_training_curves.csv` are the
actual outputs used to produce the thesis's results and figures — included so
the figures can be regenerated (`make_figures.py`) without needing to re-run
training first. Re-running `run_all_three_models.py` will overwrite them with
a fresh run.

## Setup

```bash
git clone https://github.com/nikolioudakism14/thesis-energy-aware-trajectory-forecasting.git
cd thesis-energy-aware-trajectory-forecasting
pip install -r requirements.txt
```

## Usage

1. Place the London Landing Sync CSV in `data/` (or edit `DATA_PATH` near the
   top of `run_all_three_models.py` to point at its location).
2. Train all three models and produce the result CSVs:
   ```bash
   python run_all_three_models.py
   ```
   This writes `thesis_three_model_results.csv` (summary metrics, Table 5.1)
   and `thesis_training_curves.csv` (per-epoch loss, Figure 5.1). No model's
   results are hardcoded — all three are trained fresh in this one run.
3. Generate the three thesis figures from those CSVs:
   ```bash
   python make_figures.py
   ```
   Produces `figure_5_1_training_curves.png`, `figure_5_2_energy_disaggregation.png`,
   and `figure_5_3_pareto_front.png`.

## Hardware used in the thesis

NVIDIA Tesla T4 (16GB VRAM), 2 vCPU (Intel Xeon @ 2.00GHz), 12.67GB RAM.
Energy figures were measured on this configuration via CodeCarbon 3.3.0
(GPU power via NVML, CPU via RAPL where available else load-based TDP
estimation, RAM via a static W/GB model) and will differ on other hardware —
see Sections 2.4 and 4.3–4.4 of the thesis for the full measurement
methodology and the rationale for choosing CodeCarbon over alternative
carbon-tracking frameworks.

## Scope note

These are simplified, single-layer implementations of each architecture's
core mechanism (SOFTS' series-core fusion, LSTM's recurrent gating,
Informer's ProbSparse attention), built for a controlled, like-for-like
energy comparison rather than as reproductions of the full original papers'
production-scale models. See Section 4.2.2 of the thesis for the
hyperparameter rationale and this limitation's implications.

## Citation

If you use this code, please cite the thesis: CITATION HERE.
