extends Area2D
@onready var animated_sprite_2d: AnimatedSprite2D = $AnimatedSprite2D
@onready var collected_sound: AudioStreamPlayer2D = $CollectedSound
@onready var collision_shape_2d: CollisionShape2D = $CollisionShape2D

signal apple_collected

## True once this apple has been picked up in the current episode.
## Guards against the body_entered signal double-firing.
var collected := false

func _ready() -> void:
	pass # Replace with function body.


func _process(delta: float) -> void:
	pass


func _on_body_entered(body: Node2D) -> void:
	if collected:
		return
	if body.name != "Player":
		return
	collected = true
	animated_sprite_2d.animation = "collected"
	collected_sound.play()
	collision_shape_2d.set_deferred("disabled", true)
	apple_collected.emit()

## Re-enables the collision shape; split out from reset_apple() so it can
## also be used as a standalone safety call after a deferred-call race.
## Deferred like the disable side in _on_body_entered() -- reset_apple() is
## now called synchronously from within another apple's own
## _on_body_entered() (via main.gd's _complete_level(), on both-apples
## collected), and toggling shape-monitoring state mid-physics-query-flush
## throws "Can't change this state while flushing queries" otherwise.
func force_enable_collision() -> void:
	collision_shape_2d.set_deferred("disabled", false)

## Used by RLEnvironment (via main.gd) to restart an episode without
## reloading the scene.
func reset_apple() -> void:
	collected = false
	animated_sprite_2d.animation = "default"
	force_enable_collision()
