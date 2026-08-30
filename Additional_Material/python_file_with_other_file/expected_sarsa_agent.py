"""Tabular Expected SARSA agent.

Update rule (on-policy, epsilon-greedy behavior policy pi):

    Q(s, a) += alpha * [ r + gamma * sum_a' pi(a'|s') Q(s', a') - Q(s, a) ]

No neural nets, no experience replay -- just a dict-backed Q-table keyed by
the discretized state tuples that come from Godot's rl_environment.gd.
"""

import ast
import json
import random
from collections import defaultdict

NUM_ACTIONS = 6  # idle, left, right, jump, jump_left, jump_right


class ExpectedSarsaAgent:
    def __init__(
        self,
        num_actions: int = NUM_ACTIONS,
        alpha: float = 0.1,
        gamma: float = 0.95,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.02,
        epsilon_decay_episodes: int = 2500,
        q_init: float = 5.0,
    ):
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon_start = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay_episodes = epsilon_decay_episodes
        self.epsilon = epsilon_start
        # Optimistic initialization: unvisited (state, action) pairs start
        # above what most real returns turn out to be, so an action that's
        # never been tried in a given state -- e.g. "back off right and
        # rejump" after a ceiling bump -- keeps looking attractive until
        # experience actually proves it's bad, instead of losing by default
        # to whatever action happened to be tried first.
        self.q_init = q_init
        self.q = defaultdict(lambda: [q_init] * num_actions)

    def set_epsilon_for_episode(self, episode_index: int) -> None:
        """Linear decay from epsilon_start to epsilon_end over epsilon_decay_episodes."""
        frac = min(1.0, episode_index / max(1, self.epsilon_decay_episodes))
        self.epsilon = self.epsilon_start + frac * (self.epsilon_end - self.epsilon_start)

    def action_probabilities(self, state: tuple) -> list:
        """pi(.|state) for the epsilon-greedy policy used both to act and to
        compute the expected value in the update rule.

        Ties for the max Q-value split the greedy probability mass evenly
        instead of always going to the lowest action index. With optimistic
        initialization, many under-visited states start out fully tied --
        always breaking ties toward action 0 (idle) would let a greedy
        (epsilon=0) rollout get permanently stuck idling the moment it
        lands on one, since "idle forever" is a trajectory epsilon-greedy
        training essentially never visits and so never corrects.
        """
        q_values = self.q[state]
        max_q = max(q_values)
        best_actions = [a for a, q in enumerate(q_values) if q == max_q]
        probs = [self.epsilon / self.num_actions] * self.num_actions
        bonus = (1.0 - self.epsilon) / len(best_actions)
        for a in best_actions:
            probs[a] += bonus
        return probs

    def select_action(self, state: tuple) -> int:
        probs = self.action_probabilities(state)
        if self.epsilon == 0.0:
            # Pure exploitation: which tied action wins is decided by the
            # state itself rather than a fresh coin flip, so a greedy
            # rollout reproduces the same trajectory every time instead of
            # branching whenever it re-enters a state whose Q-values are
            # still tied. Training (epsilon > 0) keeps the real random
            # draw, since that's what the Bellman target in update()
            # assumes the behaviour policy is doing.
            r = (hash(state) % 1_000_003) / 1_000_003
        else:
            r = random.random()
        cumulative = 0.0
        for a, p in enumerate(probs):
            cumulative += p
            if r <= cumulative:
                return a
        return self.num_actions - 1

    def expected_value(self, state: tuple) -> float:
        probs = self.action_probabilities(state)
        q_values = self.q[state]
        return sum(p * q for p, q in zip(probs, q_values))

    def update(self, state: tuple, action: int, reward: float, next_state: tuple, done: bool) -> None:
        current_q = self.q[state][action]
        target = reward
        if not done:
            target += self.gamma * self.expected_value(next_state)
        self.q[state][action] = current_q + self.alpha * (target - current_q)

    def save(self, path: str) -> None:
        with open(path, "w") as f:
            json.dump({str(k): v for k, v in self.q.items()}, f)

    def load(self, path: str) -> None:
        with open(path) as f:
            data = json.load(f)
        for k, v in data.items():
            self.q[ast.literal_eval(k)] = v
