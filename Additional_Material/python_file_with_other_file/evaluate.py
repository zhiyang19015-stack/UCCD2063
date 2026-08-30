"""Runs a trained Expected SARSA agent fully greedily (epsilon=0, no
learning) to check how reliably it completes the apple-pickup route.

The environment is deterministic once epsilon=0 -- same spawn, same snail
patrol timing, no randomness left in the physics -- and this checkpoint
reproduces its result across repeated evaluation runs (confirmed: two
independent 100-episode runs both scored 100/100 with matching returns).
That was not always true. Tie-breaking between equally-valued actions
(action_probabilities() splits greedy probability evenly among tied
actions; select_action() samples from that) was the first suspect for an
earlier run of run-to-run inconsistency, but forcing it to be a fixed
function of state rather than a random draw left the inconsistency rate
unchanged, which ruled it out. The actual cause was three environment
implementation bugs -- entity movement not gated to an in-progress RL
step's fixed physics-frame window, an apple's collision shape being
re-enabled with a write Godot silently rejects during a physics callback,
and a snail collision signal that could be delivered a whole episode late
-- all now fixed in rl_environment.gd/snail.gd/apple.gd/main.gd. Still
run this over many episodes rather than one or two: it costs little and
is the only way to actually confirm reliability rather than assume it.

Run AFTER train.py has produced best_q_table.json (the checkpoint saved
whenever a periodic greedy check during training succeeded -- see
train.py's docstring). Defaults to that file since it's the known-working
snapshot; q_table.json is just the raw end-of-training state and may have
drifted off a working policy by the final episode.

    python evaluate.py
    python evaluate.py --episodes 100 --q-table q_table.json
"""

import argparse
import random

from episode_runner import run_greedy_episode
from expected_sarsa_agent import ExpectedSarsaAgent
from godot_env import GodotEnv
from tcp_server import wait_for_godot

HOST = "127.0.0.1"
PORT = 5555
SEED = 42
MAX_STEPS_PER_EPISODE = 600


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5, help="how many greedy runs to check")
    parser.add_argument("--q-table", default="best_q_table.json")
    args = parser.parse_args()
    random.seed(SEED)

    agent = ExpectedSarsaAgent()
    agent.load(args.q_table)

    conn = wait_for_godot(HOST, PORT)
    env = GodotEnv(conn)

    results = []
    try:
        for i in range(1, args.episodes + 1):
            result = run_greedy_episode(env, agent, MAX_STEPS_PER_EPISODE)
            results.append(result)
            print(
                f"eval {i}/{args.episodes}  success={result['success']}  "
                f"completions={result['completions']}  steps={result['steps']:4d}  "
                f"return={result['return']:7.1f}  end={result['end_reason']}"
            )
    finally:
        env.close()
        conn.close()

    n = len(results)
    successes = sum(1 for r in results if r["success"])
    enemy_fails = sum(1 for r in results if r["end_reason"] == "enemy")
    fall_fails = sum(1 for r in results if r["end_reason"] == "fall")
    timeout_fails = sum(1 for r in results if r["end_reason"] == "timeout")
    avg_return = sum(r["return"] for r in results) / n
    avg_steps = sum(r["steps"] for r in results) / n
    avg_completions = sum(r["completions"] for r in results) / n
    rate = 100.0 * successes / n

    print(f"\nEvaluation episodes: {n}")
    print(f"Episodes completing at least one lap: {successes}")
    print(f"Completion rate: {rate:.2f}%")
    print(f"Mean laps completed per episode: {avg_completions:.2f}")
    print(f"Snail-collision failures: {enemy_fails}")
    print(f"Fall failures: {fall_fails}")
    print(f"Timeout failures (0 laps, hit step limit): {timeout_fails}")
    print(f"Average return: {avg_return:.2f}")
    print(f"Average steps: {avg_steps:.2f}")

    if 0 < successes < n:
        print(
            "\nInconsistent results across identical greedy runs -- that shouldn't "
            "happen given the environment is deterministic and epsilon=0. Tie-"
            "breaking between equally-valued actions is one possible cause, but "
            "was ruled out empirically last time this came up; see the module "
            "docstring for what actually caused it then."
        )
    elif successes == 0:
        print("\nThe greedy policy doesn't solve the level yet -- keep training (more episodes, or retune the reward shaping).")
    else:
        print("\nThe greedy policy solved the level every time.")


if __name__ == "__main__":
    main()
