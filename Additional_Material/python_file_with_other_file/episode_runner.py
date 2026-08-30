"""Shared single-episode loop, used by both train.py (learning, epsilon-
greedy) and evaluate.py (no learning, forced epsilon=0) so the two never
drift apart in how an episode is actually run."""

from expected_sarsa_agent import ExpectedSarsaAgent
from godot_env import GodotEnv
from state_discretizer import discretize


def run_episode(env: GodotEnv, agent: ExpectedSarsaAgent, max_steps: int, learn: bool) -> dict:
    raw_state, info = env.reset()
    state = discretize(raw_state)

    total_reward = 0.0
    steps = 0
    done = False

    while not done and steps < max_steps:
        action = agent.select_action(state)
        raw_next_state, reward, done, info = env.step(action)
        next_state = discretize(raw_next_state)

        if learn:
            agent.update(state, action, reward, next_state, done)

        total_reward += reward
        steps += 1
        state = next_state

    if info.get("level_completed"):
        end_reason = "success"
    elif info.get("last_death_reason"):
        end_reason = info["last_death_reason"]
    else:
        end_reason = "timeout"

    return {
        "success": bool(info.get("level_completed")),
        "apples": info.get("apples_collected"),
        "completions": info.get("completions", 0),
        "steps": steps,
        "return": total_reward,
        "end_reason": end_reason,
    }


def run_greedy_episode(env: GodotEnv, agent: ExpectedSarsaAgent, max_steps: int) -> dict:
    """Runs one episode fully greedily (epsilon=0, no Q-table updates) to
    check whether the *current* policy reliably clears the level, then
    restores the agent's exploration epsilon exactly as it was."""
    saved_epsilon = agent.epsilon
    agent.epsilon = 0.0
    try:
        return run_episode(env, agent, max_steps, learn=False)
    finally:
        agent.epsilon = saved_epsilon
