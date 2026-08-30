extends Area2D
@onready var animated_sprite_2d: AnimatedSprite2D = $AnimatedSprite2D

const SPEED = 100.0
var direction = -1.0

## When false, the snail freezes in place (used at episode end / on level
## completion so it doesn't keep patrolling into a dead/finished player).
var movement_enabled := true

## During RL training/eval, RLEnvironment toggles this true only for the
## fixed physics-frame window of an in-progress step, and false while
## waiting on the next TCP message -- see rl_environment.gd's
## _set_snails_step_active(). Left true otherwise (manual play), so
## patrol movement there is never gated.
var _step_active := true

## Emitted on overlap with the player; main.gd decides what to do with it
## (calling player.die()) rather than the snail deciding directly.
signal player_hit(body: Node2D)

var _initial_position: Vector2
var _initial_direction: float

# Called when the node enters the scene tree for the first time.
func _ready() -> void:
	_initial_position = global_position
	_initial_direction = direction

# Tied to the fixed physics step (not _process/idle) so patrol movement is
# reproducible run-to-run regardless of real-world frame timing -- matches
# player.gd's own movement, and RLEnvironment samples state on physics
# ticks, so anything driven by idle time is a hidden source of
# nondeterminism for the RL agent.
func _physics_process(delta: float) -> void:
	if movement_enabled and _step_active:
		position.x += direction * SPEED * delta

func _on_timer_timeout() -> void:
	if not movement_enabled:
		return
	direction *= -1
	animated_sprite_2d.flip_h = !animated_sprite_2d.flip_h

func _on_body_entered(body: Node2D) -> void:
	if body.name == "Player":
		player_hit.emit(body)

func set_movement_enabled(enabled: bool) -> void:
	movement_enabled = enabled

## See _step_active's doc comment. Also pauses the direction-flip Timer,
## not just position movement -- otherwise the timer keeps counting down
## on real elapsed physics ticks during the gap between steps, so a
## direction flip could still land at a different point in an episode
## from one run to the next even with position movement itself gated.
## Also disables Area2D overlap monitoring while frozen: a body_entered
## overlap detected/queued during the gap (engine-level broad-phase runs
## independently of any script's _physics_process) was observed firing
## late, on the very first step of the *next* episode, well after the
## player had teleported back to spawn -- reproduced with debug_second_episode.py.
func set_step_active(active: bool) -> void:
	_step_active = active
	var timer := get_node_or_null("Timer")
	if timer:
		timer.paused = not active
	set_deferred("monitoring", active)

## Current patrol direction (-1 left, 1 right), for use in RL state vectors.
func get_direction_value() -> float:
	return direction

## Used by RLEnvironment (via main.gd) to restart an episode without
## reloading the scene. Restores the position/direction captured on _ready.
func reset_snail() -> void:
	global_position = _initial_position
	direction = _initial_direction
	animated_sprite_2d.flip_h = direction < 0
	movement_enabled = true
	var timer := get_node_or_null("Timer")
	if timer:
		timer.start()
