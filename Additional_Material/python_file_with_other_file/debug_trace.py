"""One-off diagnostic: run one greedy episode and print every
(state, action, reward) transition, plus flag when the trajectory starts
repeating a state (a stuck cycle)."""

from collections import Counter

from expected_sarsa_agent import ExpectedSarsaAgent
from godot_env import GodotEnv
from state_discretizer import discretize
from tcp_server import wait_for_godot

ACTION_NAMES = ["idle", "left", "right", "jump", "jump_left", "jump_right"]

agent = ExpectedSarsaAgent()
agent.load("q_table.json")
agent.epsilon = 0.0

conn = wait_for_godot("127.0.0.1", 5555)
env = GodotEnv(conn)

raw_state, info = env.reset()
state = discretize(raw_state)

seen = Counter()
steps = 0
done = False
while not done and steps < 600:
    action = agent.select_action(state)
    seen[state] += 1
    if seen[state] in (2, 3, 5, 10, 20):
        print(f"step {steps:4d}  REPEAT x{seen[state]}  state={state}  action={ACTION_NAMES[action]}  q={agent.q[state]}")
    elif steps < 60 or steps % 50 == 0:
        print(f"step {steps:4d}  state={state}  action={ACTION_NAMES[action]}  q={[round(v,1) for v in agent.q[state]]}")

    raw_state, reward, done, info = env.step(action)
    state = discretize(raw_state)
    steps += 1

print(f"\nfinal info: {info}")
print(f"most common states visited:")
for s, c in seen.most_common(5):
    print(f"  x{c}  {s}  q={[round(v,1) for v in agent.q[s]]}")

env.close()
conn.close()
