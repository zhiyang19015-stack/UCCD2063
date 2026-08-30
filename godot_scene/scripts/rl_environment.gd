extends Node
## Autoload singleton. Exposes the running game as a Gym-style RL
## environment (reset() / step(action)) over tcp_bridge.gd, using a
## request/response JSON-over-TCP protocol:
##
##   {"cmd": "ping"}              -> {"cmd": "pong"}
##   {"cmd": "reset"}             -> {"cmd": "reset", "state": [...], "info": {...}}
##   {"cmd": "step", "action": n} -> {"cmd": "step", "state": [...], "reward": f, "done": bool, "info": {...}}
##   {"cmd": "state"}             -> {"cmd": "state", "state": [...], "info": {...}}
##   {"cmd": "close"}             -> {"cmd": "close", "status": "ok"}
##
## This node owns the RL-specific concerns (action space, reward shaping,
## the state vector, episode step/boundary limits). Game state itself
## (score, death reason, level-complete, resets) lives in main.gd; per-node
## reset/behaviour toggles live on player.gd/apple.gd/snail.gd. No RL logic
## is embedded in GDScript beyond this file -- state discretization and the
## Expected SARSA algorithm both live in the Python agent.
##
## Action ints (must match the Python agent's action list):
##   0 idle, 1 left, 2 right, 3 jump, 4 jump_left, 5 jump_right
##
## State vector sent to Python (18 floats, all normalized to roughly
## [-1, 1] or [0, 1]):
##   [0]  player x / viewport_w
##   [1]  player y / viewport_h
##   [2]  velocity.x normalized
##   [3]  velocity.y normalized
##   [4]  on_floor (0/1)
##   [5]  lower apple collected (0/1)
##   [6]  upper apple collected (0/1)
##   [7]  dx to current target apple / viewport_w
##   [8]  dy to current target apple / viewport_h
##   [9]  distance to current target apple / viewport_w
##   [10] dx to bridge / viewport_w
##   [11] on_bridge (0/1)
##   [12] dx to snail 1 / viewport_w
##   [13] dy to snail 1 / viewport_h
##   [14] dx to snail 2 / viewport_w
##   [15] dy to snail 2 / viewport_h
##   [16] steps this episode / MAX_STEPS
##   [17] hit_ceiling this step (0/1) -- jumped and bonked into a platform
##        edge instead of clearing it. Without this the state can't tell
##        "jumping here is blocked" apart from "jumping here is clear", so
##        the agent has no way to learn "back off and approach from a
##        different x before jumping" for spots like the underside of the
##        apple_high ledge reached from the bridge.

const VIEWPORT_W := 1280.0
const VIEWPORT_H := 720.0

## Wall-clock speed-up while an episode is running (training or eval). 1.0 is
## real time; higher just makes the demo finish sooner. A step still ends after
## a fixed ACTION_REPEAT_FRAMES physics frames and everything moves on delta, so
## per-step behaviour and the recorded results are unchanged -- only how many
## real seconds those frames take. Bump this for a faster demo (8-10 is still
## watchable); it is restored to 1.0 on "close".
const SIM_SPEED := 4.0

const ACTION_REPEAT_FRAMES := 4
const MAX_STEPS := 600
const FALL_Y_LIMIT := 780.0
const LEFT_BOUND := -50.0
const RIGHT_BOUND := 1350.0
const SNAIL_DANGER_DIST := 120.0
const BRIDGE_HALF_WIDTH := 180.0     # half the middle platform's walkable width
const BRIDGE_VERTICAL_TOLERANCE := 70.0  # must actually be standing on it, not just below/above

const STEP_PENALTY := -0.1
const APPROACH_SCALE := 0.05    # reward per px closed toward the target apple
const APPLE_REWARD := 50.0
const BRIDGE_BONUS := 10.0
const COMPLETION_REWARD := 250.0
const DEATH_PENALTY := -100.0
const TIMEOUT_PENALTY := -20.0
const DANGER_ZONE_PENALTY := -0.3
const CEILING_BUMP_PENALTY := -5.0

enum Action { IDLE, LEFT, RIGHT, JUMP, JUMP_LEFT, JUMP_RIGHT }

var _main: Node
var _player: CharacterBody2D
var _apple_low: Area2D   # larger initial y = lower on screen = eaten first
var _apple_high: Area2D
var _snails: Array = []
var _bridge_pos: Vector2

var _bound := false
var _training_active := false
var _pending_messages: Array = []

var _pending_frames := 0
var _steps := 0
var _prev_target_dist := 0.0
var _crossed_bridge := false
var _apple_low_counted := false
var _apple_high_counted := false
var _hit_ceiling_this_step := false
var _completions_counted := 0

func _ready() -> void:
	TcpBridge.message_received.connect(_on_message)

func _process(_delta: float) -> void:
	if not _bound:
		_bind_scene_nodes()
		if _bound and not _pending_messages.is_empty():
			var queued := _pending_messages.duplicate()
			_pending_messages.clear()
			for m in queued:
				_handle_message(m)

func _bind_scene_nodes() -> void:
	var scene := get_tree().current_scene
	if scene == null:
		return
	var level := scene.get_node_or_null("LevelRoot")
	if level == null:
		return

	var player := level.get_node_or_null("Player")
	var apples := level.get_node_or_null("Apple")
	var enemies := level.get_node_or_null("Enemies")
	if player == null or apples == null or enemies == null:
		return

	var apple_nodes: Array = apples.get_children()
	if apple_nodes.size() < 2:
		push_warning("[rl_environment] Expected 2 apples under LevelRoot/Apple.")
		return
	apple_nodes.sort_custom(func(a, b): return a.global_position.y < b.global_position.y)

	_main = scene
	_player = player
	_apple_high = apple_nodes[0]
	_apple_low = apple_nodes[apple_nodes.size() - 1]
	_snails = enemies.get_children()

	var bridge := level.get_node_or_null("Bridge")
	if bridge:
		_bridge_pos = bridge.global_position
	else:
		_bridge_pos = Vector2(
			(_apple_low.global_position.x + _apple_high.global_position.x) / 2.0,
			(_player.global_position.y + _apple_high.global_position.y) / 2.0
		)
		push_warning("[rl_environment] No 'Bridge' Marker2D under LevelRoot; using estimated bridge position %s. Add a Marker2D named 'Bridge' at the real bridge for a more accurate state." % str(_bridge_pos))

	_bound = true

func _physics_process(_delta: float) -> void:
	if not _training_active or _pending_frames <= 0:
		return
	_check_world_boundaries()
	if _player.is_on_ceiling():
		_hit_ceiling_this_step = true
	_pending_frames -= 1
	if _pending_frames == 0 or _main.episode_done:
		_pending_frames = 0
		_set_snails_step_active(false)
		_finish_step()

## Snails (and the player's AI-driven movement, gated in player.gd itself)
## only advance while this is true, i.e. only for the fixed
## ACTION_REPEAT_FRAMES window of an in-progress step. Real wall-clock time
## can pass between one step's response and the next request (Python
## computing the next action, TCP round-trip, OS scheduling) with no bound
## on how long, and Godot's physics loop keeps ticking through that gap
## regardless; without this gate that gap lets snails/the player drift by a
## variable extra amount each step, silently reintroducing exactly the kind
## of run-to-run nondeterminism this project otherwise goes out of its way
## to eliminate (see snail.gd/tcp_bridge.gd's physics-tick-vs-idle-time
## fixes) -- confirmed empirically: forcing the agent's own tie-breaking to
## be deterministic did not change the observed inconsistency rate at all.
func _set_snails_step_active(active: bool) -> void:
	for s in _snails:
		s.set_step_active(active)

func _on_message(msg: Dictionary) -> void:
	if not _bound:
		_pending_messages.append(msg)
		return
	_handle_message(msg)

func _handle_message(msg: Dictionary) -> void:
	match msg.get("cmd", ""):
		"ping":
			TcpBridge.send_message({"cmd": "pong"})
		"reset":
			_do_reset()
		"step":
			_start_step(int(msg.get("action", Action.IDLE)))
		"state":
			TcpBridge.send_message({"cmd": "state", "state": get_state(), "info": get_info()})
		"close":
			_do_close()

func _do_reset() -> void:
	_training_active = true
	Engine.time_scale = SIM_SPEED
	_main.set_rl_training_mode(true)
	_main.reset_episode()
	for s in _snails:
		s.set_movement_enabled(true)

	_steps = 0
	_apple_low_counted = false
	_apple_high_counted = false
	_crossed_bridge = false
	_hit_ceiling_this_step = false
	_completions_counted = 0
	_prev_target_dist = _target_distance()
	_pending_frames = 0
	_set_snails_step_active(false)

	TcpBridge.send_message({"cmd": "reset", "state": get_state(), "info": get_info()})

func _start_step(action: int) -> void:
	if _main.episode_done:
		# Caller should have reset() before stepping again; report a
		# terminal transition rather than silently advancing.
		TcpBridge.send_message({"cmd": "step", "state": get_state(), "reward": 0.0, "done": true, "info": get_info()})
		return
	_player.begin_ai_action(action)
	_hit_ceiling_this_step = false
	_pending_frames = ACTION_REPEAT_FRAMES
	_set_snails_step_active(true)

## Whether the environment is currently advancing an in-progress step's
## fixed physics-frame window. player.gd checks this (when use_ai is true)
## so its own movement is gated the same way snails are -- see
## _set_snails_step_active().
func is_step_active() -> bool:
	return _training_active and _pending_frames > 0

func _finish_step() -> void:
	_steps += 1
	var reward := STEP_PENALTY
	var done := false

	var new_completions: int = _main.total_completions - _completions_counted
	if new_completions <= 0:
		var target_dist := _target_distance()
		reward += APPROACH_SCALE * (_prev_target_dist - target_dist)
		_prev_target_dist = target_dist

		if _apple_low.collected and not _apple_low_counted:
			_apple_low_counted = true
			reward += APPLE_REWARD
		if _apple_high.collected and not _apple_high_counted:
			_apple_high_counted = true
			reward += APPLE_REWARD

		if not _crossed_bridge and _is_on_bridge():
			_crossed_bridge = true
			reward += BRIDGE_BONUS
	else:
		# Both apples were just collected and main.gd already respawned them
		# and teleported the player back to spawn this step -- skip the
		# distance-delta term (target and position just jumped) and pay the
		# completion bonus instead. Re-arm the per-cycle latches so the next
		# lap can earn apple/bridge rewards again, and reseed the distance
		# baseline from the post-teleport position/target.
		reward += COMPLETION_REWARD * new_completions
		_completions_counted = _main.total_completions
		_apple_low_counted = false
		_apple_high_counted = false
		_crossed_bridge = false
		_prev_target_dist = _target_distance()

	if _nearest_snail_distance() < SNAIL_DANGER_DIST:
		reward += DANGER_ZONE_PENALTY

	if _hit_ceiling_this_step:
		reward += CEILING_BUMP_PENALTY

	if _main.episode_done:
		reward += DEATH_PENALTY
		done = true
	elif _steps >= MAX_STEPS:
		reward += TIMEOUT_PENALTY
		done = true

	TcpBridge.send_message({"cmd": "step", "state": get_state(), "reward": reward, "done": done, "info": get_info()})

func _do_close() -> void:
	_training_active = false
	Engine.time_scale = 1.0
	_pending_frames = 0
	_set_snails_step_active(true)
	_main.set_rl_training_mode(false)
	TcpBridge.send_message({"cmd": "close", "status": "ok"})

func _check_world_boundaries() -> void:
	if not _player.alive:
		return
	var pos := _player.global_position
	if pos.y > FALL_Y_LIMIT or pos.x < LEFT_BOUND or pos.x > RIGHT_BOUND:
		_player.die("fall")

func _target_apple() -> Area2D:
	return _apple_high if _apple_low.collected else _apple_low

func _target_distance() -> float:
	return _player.global_position.distance_to(_target_apple().global_position)

func _nearest_snail_distance() -> float:
	var nearest := INF
	for s in _snails:
		var d: float = _player.global_position.distance_to(s.global_position)
		if d < nearest:
			nearest = d
	return nearest

## True only while actually standing on the bridge platform (x AND y close
## to it) -- not just anywhere below/above it at the right x.
func _is_on_bridge() -> bool:
	var p := _player.global_position
	return absf(p.x - _bridge_pos.x) < BRIDGE_HALF_WIDTH and absf(p.y - _bridge_pos.y) < BRIDGE_VERTICAL_TOLERANCE

func get_state() -> Array:
	var p := _player.global_position
	var vel: Vector2 = _player.get_normalized_velocity()
	var target := _target_apple()
	var t_delta: Vector2 = target.global_position - p
	var b_dx := _bridge_pos.x - p.x

	var s1: Vector2 = _snails[0].global_position - p if _snails.size() > 0 else Vector2.ZERO
	var s2: Vector2 = _snails[1].global_position - p if _snails.size() > 1 else Vector2.ZERO

	return [
		p.x / VIEWPORT_W,
		p.y / VIEWPORT_H,
		vel.x,
		vel.y,
		1.0 if _player.is_on_floor() else 0.0,
		1.0 if _apple_low.collected else 0.0,
		1.0 if _apple_high.collected else 0.0,
		t_delta.x / VIEWPORT_W,
		t_delta.y / VIEWPORT_H,
		t_delta.length() / VIEWPORT_W,
		b_dx / VIEWPORT_W,
		1.0 if _is_on_bridge() else 0.0,
		s1.x / VIEWPORT_W,
		s1.y / VIEWPORT_H,
		s2.x / VIEWPORT_W,
		s2.y / VIEWPORT_H,
		float(_steps) / float(MAX_STEPS),
		1.0 if _hit_ceiling_this_step else 0.0,
	]

func get_info() -> Dictionary:
	return {
		"apples_collected": _main.apples_collected,
		"steps": _steps,
		"phase": 2 if (_apple_low.collected and _apple_high.collected) else (1 if _apple_low.collected else 0),
		"level_completed": _main.level_completed,
		"completions": _main.total_completions,
		"last_death_reason": _main.last_death_reason,
	}

func is_training_mode() -> bool:
	return _training_active
