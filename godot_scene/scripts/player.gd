extends CharacterBody2D
@onready var animated_sprite_2d: AnimatedSprite2D = $AnimatedSprite2D
@onready var jump_sound: AudioStreamPlayer2D = $jumpSound
@onready var death_sound: AudioStreamPlayer2D = $DeathSound

const SPEED = 300.0
const JUMP_VELOCITY = -850.0

var alive = true

## AI control (driven by RLEnvironment during training). When use_ai is
## true, keyboard input is ignored and movement instead follows
## ai_direction / ai_jump_pulse, set via begin_ai_action().
var use_ai := false
var ai_action := 0
var ai_direction := 0.0
var ai_jump_pulse := false

signal player_died(reason: String)

func _physics_process(delta: float) -> void:

	if !alive:
		return

	# During RL training/eval, freeze entirely between steps -- otherwise
	# gravity/movement keep advancing on real elapsed physics ticks while
	# waiting on the next TCP message from Python, which has no fixed
	# duration, silently reintroducing run-to-run drift. Mirrors
	# snail.gd's own step-gating; see rl_environment.gd's
	# _set_snails_step_active() for the full rationale.
	if use_ai and not RLEnvironment.is_step_active():
		return

	# Add animation
	if velocity.x > 1 or velocity.x < -1:
		animated_sprite_2d.animation = "running"
	else:
		animated_sprite_2d.animation = "idle"

	# Add the gravity.
	if not is_on_floor():
		velocity += get_gravity() * delta
		animated_sprite_2d.animation = "jumping"

	var direction: float
	var jump_requested: bool
	if use_ai:
		direction = ai_direction
		jump_requested = ai_jump_pulse
		ai_jump_pulse = false
	else:
		# As good practice, you should replace UI actions with custom gameplay actions.
		direction = Input.get_axis("left", "right")
		jump_requested = Input.is_action_just_pressed("jump")

	# Handle jump.
	if jump_requested and is_on_floor():
		velocity.y = JUMP_VELOCITY
		jump_sound.play()

	# Get the input direction and handle the movement/deceleration.
	if direction:
		velocity.x = direction * SPEED
	else:
		velocity.x = move_toward(velocity.x, 0, SPEED)

	move_and_slide()

	# Animation direction
	if direction == 1.0:
		animated_sprite_2d.flip_h = false
	elif direction == -1.0:
		animated_sprite_2d.flip_h = true

func die(reason: String = "enemy") -> void:
	if not alive:
		return
	death_sound.play()
	animated_sprite_2d.animation = "dying"
	alive = false
	player_died.emit(reason)

## Used by RLEnvironment to restart an episode without reloading the scene.
func reset_player(new_position: Vector2) -> void:
	alive = true
	global_position = new_position
	velocity = Vector2.ZERO
	ai_action = 0
	ai_direction = 0.0
	ai_jump_pulse = false
	animated_sprite_2d.animation = "idle"

## Starts one discrete RL action (0 idle, 1 left, 2 right, 3 jump,
## 4 jump_left, 5 jump_right). Call once per RL-step; the resulting
## direction is held every physics frame automatically, while the jump
## impulse fires exactly once (mirrors Input.is_action_just_pressed).
func begin_ai_action(action: int) -> void:
	ai_action = action
	match action:
		1, 4:
			ai_direction = -1.0
		2, 5:
			ai_direction = 1.0
		_:
			ai_direction = 0.0
	ai_jump_pulse = action == 3 or action == 4 or action == 5

## Velocity scaled to roughly [-1, 1], for use in RL state vectors.
func get_normalized_velocity() -> Vector2:
	return Vector2(velocity.x / SPEED, velocity.y / absf(JUMP_VELOCITY))
