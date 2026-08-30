"""Converts the raw 18-dim continuous state Godot's rl_environment.gd sends
into the coarse (direction, distance) bins the tabular Expected-SARSA agent
keys its Q-table on.

Keeping this in Python (not GDScript) means the environment stays a
generic, continuous-state Gym-style env -- usable by other algorithms too
-- while the binning scheme specific to *this* tabular agent can be tuned
without touching the Godot project at all.

Raw state layout (see rl_environment.gd's get_state() docstring):
  0 player_x_norm      6 apple_high_collected   12 snail1_dx_norm
  1 player_y_norm      7 target_dx_norm         13 snail1_dy_norm
  2 vel_x_norm         8 target_dy_norm         14 snail2_dx_norm
  3 vel_y_norm         9 target_dist_norm       15 snail2_dy_norm
  4 on_floor           10 bridge_dx_norm        16 steps_ratio
  5 apple_low_collected 11 on_bridge            17 hit_ceiling

hit_ceiling is included as its own bin (not folded into anything else):
without it, a spot where jumping is physically blocked (e.g. straight up
from the bridge, under the apple_high ledge) looks identical in state-space
to a spot where the same jump is clear, so the agent has no way to learn
"back off sideways and re-approach" for blocked spots specifically.

vel_x/vel_y are intentionally NOT binned in. It was tried (3x3 vel bins),
which grew the table ~9x (~11.6k -> ~105k states) and, under the same
3000-episode budget, converged to a *worse* policy (1 success in 3000
training episodes vs. many with this smaller state space, and a greedy
policy that confidently walked into the snail every single evaluation
run instead of solving the level) -- confirmed empirically, not assumed.
Revisit only alongside a much larger training budget.
"""

import math

VIEWPORT_W = 1280.0
VIEWPORT_H = 720.0

ALIGN_THRESHOLD_PX = 20.0
NEAR_DIST_PX = 150.0
MID_DIST_PX = 400.0
SNAIL_DANGER_DIST_PX = 120.0


def _dir_bin(delta_px: float, threshold: float = ALIGN_THRESHOLD_PX) -> int:
    if delta_px < -threshold:
        return -1
    if delta_px > threshold:
        return 1
    return 0


def _dist_bin(dist_px: float) -> int:
    if dist_px < NEAR_DIST_PX:
        return 0
    if dist_px < MID_DIST_PX:
        return 1
    return 2


def discretize(raw_state: list) -> tuple:
    """Returns a hashable tuple:
    (phase, target_dx, target_dy, target_dist, bridge_dx, on_bridge,
     snail_dx, snail_dist, on_floor, hit_ceiling)
    """
    (
        _px, _py, _vx, _vy, on_floor,
        apple_low, apple_high,
        t_dx_n, t_dy_n, t_dist_n,
        b_dx_n, on_bridge,
        s1_dx_n, s1_dy_n, s2_dx_n, s2_dy_n,
        _steps_ratio,
        hit_ceiling,
    ) = raw_state

    phase = 2 if (apple_low and apple_high) else (1 if apple_low else 0)

    t_dx = t_dx_n * VIEWPORT_W
    t_dy = t_dy_n * VIEWPORT_H
    t_dist = t_dist_n * VIEWPORT_W
    b_dx = b_dx_n * VIEWPORT_W

    s1_dx, s1_dy = s1_dx_n * VIEWPORT_W, s1_dy_n * VIEWPORT_H
    s2_dx, s2_dy = s2_dx_n * VIEWPORT_W, s2_dy_n * VIEWPORT_H
    d1 = math.hypot(s1_dx, s1_dy)
    d2 = math.hypot(s2_dx, s2_dy)
    nearest_dx, nearest_dist = (s1_dx, d1) if d1 <= d2 else (s2_dx, d2)

    return (
        phase,
        _dir_bin(t_dx),
        _dir_bin(t_dy),
        _dist_bin(t_dist),
        _dir_bin(b_dx),
        int(on_bridge),
        _dir_bin(nearest_dx),
        0 if nearest_dist < SNAIL_DANGER_DIST_PX else 1,
        int(on_floor),
        int(hit_ceiling),
    )
