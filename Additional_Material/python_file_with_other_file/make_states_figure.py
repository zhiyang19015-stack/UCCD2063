"""Generates Figure 4.4 (distinct states visited over training) from a
training_log.csv that has the states_visited column (added specifically for
this figure -- the originally reported v7 run predates this column, so this
uses a separate instrumented run just for this one curve's shape). Run from
python_agent/:

    python make_states_figure.py
"""

import csv
import os

import matplotlib.pyplot as plt

LOG_PATH = "training_log.csv"
OUT_DIR = "figures"

os.makedirs(OUT_DIR, exist_ok=True)

episodes, states_visited = [], []
with open(LOG_PATH, newline="") as f:
    for row in csv.DictReader(f):
        episodes.append(int(row["episode"]))
        states_visited.append(int(row["states_visited"]))

plt.figure(figsize=(8, 4.5))
plt.plot(episodes, states_visited, linewidth=1.8, color="#8172B2")
plt.xlabel("Episode")
plt.ylabel("Distinct states visited (cumulative)")
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "fig_4_4_states.png"), dpi=150)
plt.close()

print(f"Wrote fig_4_4_states.png -- final count: {states_visited[-1]} states")
