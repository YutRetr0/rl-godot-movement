extends Node3D

const MSG_RESET := 1
const MSG_STEP := 2
const MAX_SPAWN_ATTEMPTS := 16

@export var bind_host: String = "127.0.0.1"
@export var port: int = 9000
@export var frame_skip: int = 3
@export var max_steps: int = 300
@export var success_radius: float = 1.2
@export var success_bonus: float = 10.0
@export var time_penalty: float = -0.01
@export var collision_penalty: float = -0.05
@export var arena_half_extent_x: float = 10.0
@export var arena_half_extent_z: float = 10.0
@export var max_speed: float = 6.0
@export var max_vertical_speed: float = 4.0
@export var max_yaw_rate: float = 2.5
@export var spawn_height: float = 1.0
@export var target_height: float = 0.5
@export var spawn_margin: float = 1.5
@export var min_spawn_target_distance: float = 3.0

@onready var agent: CharacterBody3D = $Agent
@onready var target: Node3D = $Target

var rng := RandomNumberGenerator.new()
var tcp_server := TCPServer.new()
var peer: StreamPeerTCP = null
var recv_buffer := PackedByteArray()
var command_queue: Array[Dictionary] = []

var stepping := false
var step_frames_left := 0
var current_action := Vector4.ZERO
var previous_distance := 0.0
var collided_during_step := false
var episode_steps := 0

func _ready() -> void:
    _apply_cmdline_overrides()
    _setup_agent()
    var listen_err := tcp_server.listen(port, bind_host)
    if listen_err != OK:
        push_error("Failed to start TCP server on %s:%d (%d)" % [bind_host, port, listen_err])
    else:
        print("RL server listening on %s:%d" % [bind_host, port])
    _reset_episode(null)


func _exit_tree() -> void:
    _disconnect_client("Shutting down server")
    if tcp_server.is_listening():
        tcp_server.stop()


func _process(_delta: float) -> void:
    _accept_client_if_needed()
    _poll_client()
    _parse_messages()
    _dispatch_next_command()


func _physics_process(_delta: float) -> void:
    if not stepping:
        return

    _apply_action_once(1.0 / float(Engine.physics_ticks_per_second))
    step_frames_left -= 1

    if step_frames_left > 0:
        return

    episode_steps += 1
    var current_distance := _distance_to_target()
    var reward := previous_distance - current_distance + time_penalty
    if collided_during_step:
        reward += collision_penalty

    var success := current_distance <= success_radius
    if success:
        reward += success_bonus

    var timeout := episode_steps >= max_steps
    var done := success or timeout

    _send_response(_build_observation(), reward, done)
    stepping = false


func _setup_agent() -> void:
    agent.motion_mode = CharacterBody3D.MOTION_MODE_FLOATING
    agent.velocity = Vector3.ZERO


func _apply_cmdline_overrides() -> void:
    for arg in OS.get_cmdline_user_args():
        if arg.begins_with("--port="):
            var arg_port := arg.get_slice("=", 1)
            if arg_port.is_valid_int():
                port = int(arg_port)
        elif arg.begins_with("--frame-skip="):
            var arg_frame_skip := arg.get_slice("=", 1)
            if arg_frame_skip.is_valid_int():
                frame_skip = max(1, int(arg_frame_skip))
        elif arg.begins_with("--max-steps="):
            var arg_max_steps := arg.get_slice("=", 1)
            if arg_max_steps.is_valid_int():
                max_steps = max(1, int(arg_max_steps))


func _accept_client_if_needed() -> void:
    if peer != null:
        return
    if not tcp_server.is_connection_available():
        return

    peer = tcp_server.take_connection()
    if peer != null:
        peer.set_no_delay(true)
        recv_buffer = PackedByteArray()
        command_queue.clear()
        stepping = false
        print("Client connected")


func _poll_client() -> void:
    if peer == null:
        return

    peer.poll()
    if peer.get_status() != StreamPeerTCP.STATUS_CONNECTED:
        _disconnect_client("Client disconnected")
        return

    var available := peer.get_available_bytes()
    if available <= 0:
        return

    while available > 0:
        var read_size := min(available, 4096)
        var read_result := peer.get_partial_data(read_size)
        var err: int = read_result[0]
        var chunk: PackedByteArray = read_result[1]
        if err != OK and err != ERR_BUSY:
            _disconnect_client("Socket read error: %d" % err)
            return
        if chunk.is_empty():
            break
        recv_buffer.append_array(chunk)
        available -= chunk.size()


func _parse_messages() -> void:
    while true:
        if recv_buffer.size() < 1:
            return

        var msg_type := recv_buffer[0]
        if msg_type == MSG_RESET:
            if recv_buffer.size() < 2:
                return

            var has_seed := recv_buffer[1]
            if has_seed > 1:
                _disconnect_client("Invalid RESET seed flag")
                return

            var needed := 2 + (4 if has_seed == 1 else 0)
            if recv_buffer.size() < needed:
                return

            var parser := StreamPeerBuffer.new()
            parser.big_endian = false
            parser.data_array = recv_buffer.slice(0, needed)
            parser.get_u8() # type
            parser.get_u8() # has_seed
            var seed: Variant = null
            if has_seed == 1:
                seed = parser.get_32()

            _consume_buffer(needed)
            command_queue.push_back({"type": MSG_RESET, "seed": seed})
        elif msg_type == MSG_STEP:
            var step_size := 1 + 16
            if recv_buffer.size() < step_size:
                return

            var parser_step := StreamPeerBuffer.new()
            parser_step.big_endian = false
            parser_step.data_array = recv_buffer.slice(0, step_size)
            parser_step.get_u8() # type
            var action := Vector4(
                parser_step.get_float(),
                parser_step.get_float(),
                parser_step.get_float(),
                parser_step.get_float()
            )

            _consume_buffer(step_size)
            command_queue.push_back({"type": MSG_STEP, "action": action})
        else:
            _disconnect_client("Unknown message type: %d" % msg_type)
            return


func _consume_buffer(bytes_to_consume: int) -> void:
    if bytes_to_consume >= recv_buffer.size():
        recv_buffer = PackedByteArray()
    else:
        recv_buffer = recv_buffer.slice(bytes_to_consume)


func _dispatch_next_command() -> void:
    if peer == null:
        return
    if stepping:
        return
    if command_queue.is_empty():
        return

    var command: Dictionary = command_queue.pop_front()
    if command["type"] == MSG_RESET:
        _reset_episode(command["seed"])
        _send_response(_build_observation(), 0.0, false)
    elif command["type"] == MSG_STEP:
        current_action = command["action"]
        previous_distance = _distance_to_target()
        collided_during_step = false
        step_frames_left = max(1, frame_skip)
        stepping = true


func _apply_action_once(delta: float) -> void:
    agent.rotate_y(current_action.w * max_yaw_rate * delta)

    var local_movement_direction := Vector3(current_action.x, current_action.y, current_action.z)
    if local_movement_direction.length() > 1.0:
        local_movement_direction = local_movement_direction.normalized()

    var world_direction := agent.global_transform.basis * local_movement_direction
    agent.velocity = Vector3(
        world_direction.x * max_speed,
        local_movement_direction.y * max_vertical_speed,
        world_direction.z * max_speed
    )
    agent.move_and_slide()

    if agent.get_slide_collision_count() > 0:
        collided_during_step = true


func _reset_episode(seed: Variant) -> void:
    if seed != null:
        var normalized_seed := int(seed)
        if normalized_seed < 0:
            normalized_seed += 4294967296
        rng.seed = normalized_seed
    else:
        rng.randomize()

    episode_steps = 0
    stepping = false
    step_frames_left = 0
    collided_during_step = false
    agent.velocity = Vector3.ZERO
    agent.global_position = _random_spawn_position(spawn_height)
    agent.rotation = Vector3(0.0, rng.randf_range(-PI, PI), 0.0)

    var attempts := 0
    while true:
        target.global_position = _random_spawn_position(target_height)
        if target.global_position.distance_to(agent.global_position) >= min_spawn_target_distance:
            break
        attempts += 1
        if attempts > MAX_SPAWN_ATTEMPTS:
            break


func _random_spawn_position(height: float) -> Vector3:
    return Vector3(
        rng.randf_range(-arena_half_extent_x + spawn_margin, arena_half_extent_x - spawn_margin),
        height,
        rng.randf_range(-arena_half_extent_z + spawn_margin, arena_half_extent_z - spawn_margin)
    )


func _distance_to_target() -> float:
    return agent.global_position.distance_to(target.global_position)


func _build_observation() -> PackedFloat32Array:
    var relative_world := target.global_position - agent.global_position
    var local_basis := agent.global_transform.basis.inverse()
    var relative_local := local_basis * relative_world
    var local_velocity := local_basis * agent.velocity
    var yaw := agent.rotation.y

    return PackedFloat32Array([
        relative_local.x,
        relative_local.y,
        relative_local.z,
        local_velocity.x,
        local_velocity.y,
        local_velocity.z,
        sin(yaw),
        cos(yaw)
    ])


func _send_response(observation: PackedFloat32Array, reward: float, done: bool) -> void:
    if peer == null:
        return

    var writer := StreamPeerBuffer.new()
    writer.big_endian = false
    writer.put_32(observation.size())
    for value in observation:
        writer.put_float(value)
    writer.put_float(reward)
    writer.put_u8(1 if done else 0)

    var send_err := peer.put_data(writer.data_array)
    if send_err != OK:
        _disconnect_client("Socket write error: %d" % send_err)


func _disconnect_client(reason: String) -> void:
    if peer != null:
        print(reason)
        peer.disconnect_from_host()
    peer = null
    recv_buffer = PackedByteArray()
    command_queue.clear()
    stepping = false
