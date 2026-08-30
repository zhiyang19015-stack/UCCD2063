"""Generates the training figures (reward, steps, epsilon curves) from the
real training_log.csv produced by train.py. Run from python_agent/:

    python make_training_figures.py

Saves PNGs into python_agent/figures/.
"""

import csv
import os

import matplotlib.pyplot as plt

LOG_PATH = "training_log.csv"
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

episodes, returns, steps, epsilons = [], [], [], []
with open(LOG_PATH, newline="") as f:
    for row in csv.DictReader(f):
        episodes.append(int(row["episode"]))
        returns.append(float(row["return"]))
        steps.append(int(row["steps"]))
        epsilons.append(float(row["epsilon"]))


def rolling_mean(values, window=50):
    out = []
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out.append(sum(values[lo : i + 1]) / (i - lo + 1))
    return out


# Figure 1: total reward per episode
plt.figure(figsize=(8, 4.5))
plt.plot(episodes, returns, linewidth=0.4, alpha=0.35, color="#4C72B0", label="per-episode return")
plt.plot(episodes, rolling_mean(returns), linewidth=1.8, color="#0B3C7A", label="50-episode rolling mean")
plt.axhline(0, color="gray", linewidth=0.7, linestyle="--")
plt.xlabel("Episode")
plt.ylabel("Total reward (return)")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4_1_reward.png"), dpi=150)
plt.close()

# Figure 2: steps per episode
plt.figure(figsize=(8, 4.5))
plt.plot(episodes, steps, linewidth=0.4, alpha=0.35, color="#55A868", label="per-episode steps")
plt.plot(episodes, rolling_mean(steps), linewidth=1.8, color="#1B5E20", label="50-episode rolling mean")
plt.xlabel("Episode")
plt.ylabel("Steps taken")
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4_2_steps.png"), dpi=150)
plt.close()

# Figure 3: epsilon decay
plt.figure(figsize=(8, 4.5))
plt.plot(episodes, epsilons, linewidth=1.8, color="#C44E52")
plt.xlabel("Episode")
plt.ylabel("Epsilon (exploration rate)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4_3_epsilon.png"), dpi=150)
plt.close()

print(f"Wrote {len(episodes)} episodes' worth of figures to {OUT_DIR}/")
print("  fig_4_1_reward.png")
print("  fig_4_2_steps.png")
print("  fig_4_3_epsilon.png")
