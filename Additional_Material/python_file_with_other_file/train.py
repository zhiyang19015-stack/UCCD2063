"""Trains a tabular Expected SARSA agent to play the Godot level over TCP.

Run this FIRST (it listens and waits), then run/play the Godot project --
rl_environment.gd will connect to it automatically and training starts.

    python train.py

Q-table is checkpointed to q_table.json every 100 episodes and at the end,
alongside training_state.json (just the last completed episode number).
Per-episode return/length/epsilon is logged to training_log.csv.

Resuming: if q_table.json and training_state.json already exist (e.g. this
script was stopped and restarted), training picks up from the next episode
using the loaded Q-table and the correct point in the epsilon decay
schedule, instead of starting over from scratch. Delete both files (or
point Q_TABLE_PATH/STATE_PATH elsewhere) for a genuinely fresh run.

The environment is deterministic once epsilon=0 (fixed spawn, fixed snail
patrol timing, no randomness left in action selection), so "does the
agent collect both apples every run" reduces to "has the greedy policy
converged to a working route" -- it either always succeeds or always
fails the same way, it can't succeed 9/10 times. Every GREEDY_CHECK_EVERY
episodes this script pauses training and runs ONE fully-greedy (epsilon=0,
no learning) episode to report whether that's true yet; run evaluate.py
afterwards to confirm across several runs before treating it as solved.

Policy quality is NOT monotonic over training -- a greedy check can
succeed at episode 2400 and then fail again by 3000 as later updates
perturb it. So every time a greedy check succeeds, the Q-table is also
snapshotted to best_q_table.json (+ best_checkpoint_info.json with the
episode/steps/return it achieved). That snapshot is only ever overwritten
by a *later* success, never by a failure, so it's always a known-working
policy even if the final q_table.json has since drifted off it. Point
evaluate.py --q-table best_q_table.json at it to confirm.
"""

import argparse
import csv
import json
import os
import random

from episode_runner import run_episode, run_greedy_episode
from expected_sarsa_agent import ExpectedSarsaAgent
from godot_env import GodotEnv
from tcp_server import wait_for_godot

HOST = "127.0.0.1"
PORT = 5555
SEED = 42
NUM_EPISODES = 3000
MAX_STEPS_PER_EPISODE = 600  # safety net; Godot enforces the same cap itself
CHECKPOINT_EVERY = 100
GREEDY_CHECK_EVERY = 200
Q_TABLE_PATH = "q_table.json"
STATE_PATH = "training_state.json"
LOG_PATH = "training_log.csv"
BEST_Q_TABLE_PATH = "best_q_table.json"
BEST_INFO_PATH = "best_checkpoint_info.json"


def _parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gamma", type=float, default=0.95, help="discount factor")
    parser.add_argument("--q-table", default=Q_TABLE_PATH)
    parser.add_argument("--state", default=STATE_PATH)
    parser.add_argument("--log", default=LOG_PATH)
    parser.add_argument("--best-q-table", default=BEST_Q_TABLE_PATH)
    parser.add_argument("--best-info", default=BEST_INFO_PATH)
    return parser.parse_args()


def _save_checkpoint(agent: ExpectedSarsaAgent, episode: int) -> None:
    agent.save(Q_TABLE_PATH)
    with open(STATE_PATH, "w") as f:
        json.dump({"last_completed_episode": episode}, f)


def _save_best_checkpoint(agent: ExpectedSarsaAgent, episode: int, check: dict) -> None:
    agent.save(BEST_Q_TABLE_PATH)
    with open(BEST_INFO_PATH, "w") as f:
        json.dump({
            "episode": episode,
            "steps": check["steps"],
            "return": check["return"],
        }, f)
    print(f"  -> new best checkpoint saved to {BEST_Q_TABLE_PATH} (episode {episode}, {check['steps']} steps)")


def main() -> None:
    args = _parse_args()
    global Q_TABLE_PATH, STATE_PATH, LOG_PATH, BEST_Q_TABLE_PATH, BEST_INFO_PATH
    Q_TABLE_PATH, STATE_PATH, LOG_PATH = args.q_table, args.state, args.log
    BEST_Q_TABLE_PATH, BEST_INFO_PATH = args.best_q_table, args.best_info
    random.seed(SEED)

    conn = wait_for_godot(HOST, PORT)
    env = GodotEnv(conn)
    agent = ExpectedSarsaAgent(epsilon_decay_episodes=1800, gamma=args.gamma)

    start_episode = 1
    log_rows = []
    if os.path.exists(Q_TABLE_PATH) and os.path.exists(STATE_PATH):
        agent.load(Q_TABLE_PATH)
        with open(STATE_PATH) as f:
            start_episode = json.load(f)["last_completed_episode"] + 1
        if os.path.exists(LOG_PATH):
            with open(LOG_PATH, newline="") as f:
                log_rows = [tuple(row) for row in csv.reader(f)][1:]  # drop header
        print(f"Resuming from episode {start_episode} using {Q_TABLE_PATH} ({len(agent.q)} states learned so far).")

    if start_episode > NUM_EPISODES:
        print(f"Checkpoint already reached episode {start_episode - 1} >= NUM_EPISODES={NUM_EPISODES}; nothing to do.")
        env.close()
        conn.close()
        return

    last_completed_episode = start_episode - 1
    try:
        for episode in range(start_episode, NUM_EPISODES + 1):
            agent.set_epsilon_for_episode(episode - 1)

            result = run_episode(env, agent, MAX_STEPS_PER_EPISODE, learn=True)
            last_completed_episode = episode

            log_rows.append((
                episode, round(result["return"], 2), result["steps"],
                round(agent.epsilon, 4), result["end_reason"], len(agent.q),
            ))
            print(
                f"episode {episode}/{NUM_EPISODES}  "
                f"return={result['return']:7.1f}  steps={result['steps']:4d}  epsilon={agent.epsilon:.3f}  "
                f"apples={result['apples']}  end={result['end_reason']}"
            )

            if episode % CHECKPOINT_EVERY == 0:
                _save_checkpoint(agent, episode)

            if episode % GREEDY_CHECK_EVERY == 0:
                check = run_greedy_episode(env, agent, MAX_STEPS_PER_EPISODE)
                print(
                    f"  [greedy check @ {episode}] success={check['success']}  "
                    f"apples={check['apples']}  steps={check['steps']:4d}  end={check['end_reason']}"
                )
                if check["success"]:
                    _save_best_checkpoint(agent, episode, check)
    finally:
        _save_checkpoint(agent, last_completed_episode)
        with open(LOG_PATH, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["episode", "return", "steps", "epsilon", "end_reason", "states_visited"])
            writer.writerows(log_rows)
        env.close()
        conn.close()

    print(f"Training complete. Q-table -> {Q_TABLE_PATH}, log -> {LOG_PATH}")
    if os.path.exists(BEST_Q_TABLE_PATH):
        with open(BEST_INFO_PATH) as f:
            best = json.load(f)
        print(
            f"Best known-working checkpoint -> {BEST_Q_TABLE_PATH} "
            f"(from episode {best['episode']}, {best['steps']} steps). "
            f"Run: python evaluate.py --q-table {BEST_Q_TABLE_PATH}"
        )
    else:
        print("No greedy check ever succeeded during this run -- no best checkpoint saved.")
    print("Run evaluate.py to check the greedy policy across several runs.")


if __name__ == "__main__":
    main()
