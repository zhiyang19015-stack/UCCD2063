extends Node
## Autoload singleton. Pure TCP transport between Godot and the Python
## Expected-SARSA agent. Godot is the TCP client, Python is the TCP server.
## Protocol: newline-delimited JSON, one message per line.
##
## This node knows nothing about the game -- rl_environment.gd owns all
## game/RL logic and only talks to this node via send_message() / the
## message_received signal.

const HOST := "127.0.0.1"
const PORT := 5555
const RECONNECT_INTERVAL := 1.0

var _peer := StreamPeerTCP.new()
var _is_connected := false
var _reconnect_timer := 0.0
var _read_buffer := PackedByteArray()

signal message_received(msg: Dictionary)
signal connected_to_agent

func _ready() -> void:
	_try_connect()

func _process(delta: float) -> void:
	_peer.poll()
	var status := _peer.get_status()

	if status == StreamPeerTCP.STATUS_CONNECTED:
		if not _is_connected:
			_is_connected = true
			print("[tcp_bridge] Connected to Python agent at %s:%d" % [HOST, PORT])
			connected_to_agent.emit()
		_read_available()
	else:
		if _is_connected:
			print("[tcp_bridge] Lost connection to Python agent.")
		_is_connected = false
		if status != StreamPeerTCP.STATUS_CONNECTING:
			_reconnect_timer -= delta
			if _reconnect_timer <= 0.0:
				_try_connect()

func _try_connect() -> void:
	_reconnect_timer = RECONNECT_INTERVAL
	if _peer.get_status() != StreamPeerTCP.STATUS_NONE:
		_peer.disconnect_from_host()
	_peer.connect_to_host(HOST, PORT)

func is_connected_to_agent() -> bool:
	return _is_connected

func send_message(data: Dictionary) -> void:
	if not _is_connected:
		return
	var line := JSON.stringify(data) + "\n"
	_peer.put_data(line.to_utf8_buffer())

func _read_available() -> void:
	var n := _peer.get_available_bytes()
	if n <= 0:
		return
	var chunk = _peer.get_data(n)
	if chunk[0] != OK:
		return
	_read_buffer.append_array(chunk[1])
	_extract_lines()

func _extract_lines() -> void:
	while true:
		var newline_index := _read_buffer.find(10) # ASCII '\n'
		if newline_index < 0:
			break
		var line_bytes := _read_buffer.slice(0, newline_index)
		_read_buffer = _read_buffer.slice(newline_index + 1)
		if line_bytes.is_empty():
			continue
		var line_text := line_bytes.get_string_from_utf8()
		var parsed = JSON.parse_string(line_text)
		if typeof(parsed) == TYPE_DICTIONARY:
			message_received.emit(parsed)
