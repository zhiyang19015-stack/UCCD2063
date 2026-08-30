extends Node2D
## Owns game-level state and signal wiring. RLEnvironment (rl_environment.gd)
## drives episodes through reset_episode() / set_rl_training_mode() and reads
## apples_collected / level_completed / episode_done / last_death_reason
## rather than touching Player/Apple/Snail nodes directly.

var apples_collected := 0
var level_completed := false
var total_completions := 0
var episode_done := false
var last_death_reason := ""
var training_mode := false

var _player: CharacterBody2D
var _apples: Array = []
var _snails: Array = []
var _spawn_position: Vector2

@onready var _ui: CanvasLayer = $UI
@onready var _reset_button: Button = $UI/ResetButton
@onready var _completion_label: Label = $UI/CompletionLabel

func _ready() -> void:
	_setup_level()
	_reset_button.pressed.connect(_on_reset_button_pressed)
	_completion_label.visible = false

func _setup_level() -> void:
	var level := $LevelRoot
	_player = level.get_node("Player")
	_spawn_position = _player.global_position
	_player.player_died.connect(_on_player_died)

	var apple_container := level.get_node("Apple")
	_apples = apple_container.get_children()
	for apple in _apples:
		apple.apple_collected.connect(_on_apple_collected)

	var enemies := level.get_node_or_null("Enemies")
	if enemies:
		_snails = enemies.get_children()
		for snail in _snails:
			snail.player_hit.connect(_on_player_hit)

func _on_apple_collected() -> void:
	apples_collected += 1
	if apples_collected >= _apples.size():
		_complete_level()

func _on_player_hit(body: Node2D) -> void:
	if not body.alive:
		return
	# An Area2D overlap detected right at the edge of a step's frozen
	# window can have its body_entered signal delivered a frame -- or, it
	# turns out, a whole episode -- late: observed landing on the very
	# first step of the *next* episode, with the player already
	# teleported back to spawn, nowhere near any snail. is_step_active()
	# alone doesn't catch this, since by the time it's delivered the new
	# episode's own first step is legitimately active. Re-validate against
	# the player's actual current position instead of trusting a queued
	# signal to still be accurate when it arrives.
	var near_snail := false
	for snail in _snails:
		if body.global_position.distance_to(snail.global_position) < 40.0:
			near_snail = true
			break
	if not near_snail:
		return
	body.die("enemy")

func _on_player_died(reason: String) -> void:
	last_death_reason = reason
	episode_done = true
	for snail in _snails:
		snail.set_movement_enabled(false)

## Fires each time both apples have been collected. Respawns the apples and
## returns the player to the spawn point so the pickup cycle repeats instead
## of ending the run -- in both manual play and RL training/eval.
## level_completed is a sticky "completed at least once this episode" flag
## (what success reporting reads); total_completions is the actual count,
## since more than one completion can happen inside a single episode --
## RLEnvironment diffs it each step to pay COMPLETION_REWARD per cycle
## without treating a completion as episode-ending.
func _complete_level() -> void:
	level_completed = true
	total_completions += 1
	apples_collected = 0
	for apple in _apples:
		apple.reset_apple()
	_player.reset_player(_spawn_position)
	if training_mode:
		return
	_completion_label.text = "Level Complete!"
	_completion_label.visible = true
	get_tree().create_timer(1.5).timeout.connect(func(): _completion_label.visible = false)

## Resets player, apples, enemies and episode bookkeeping without reloading
## the scene -- this is what makes frequent RL episode resets cheap.
func reset_episode() -> void:
	apples_collected = 0
	level_completed = false
	total_completions = 0
	episode_done = false
	last_death_reason = ""
	_completion_label.visible = false

	_player.reset_player(_spawn_position)
	for apple in _apples:
		apple.reset_apple()
	for snail in _snails:
		snail.reset_snail()

## Switches the player between keyboard control and RLEnvironment-driven
## control, and hides the manual-play UI while training.
func set_rl_training_mode(enabled: bool) -> void:
	training_mode = enabled
	_player.use_ai = enabled
	_ui.visible = not enabled

func _on_reset_button_pressed() -> void:
	reset_episode()
