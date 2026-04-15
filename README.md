# rl-godot-movement

Minimal Godot 4.x RL demo for 3D movement/navigation with a TCP API and Python PPO harness.

## Target engine version

- **Godot 4.2.2.stable** (project targets Godot 4.2.x)

## Project layout

- `project.godot` - Godot project file with fixed physics tick defaults.
- `scenes/Arena.tscn` - Minimal 3D arena scene.
- `scripts/Arena.gd` - RL environment logic + TCP server.
- `python/env_client.py` - low-level TCP client.
- `python/gym_env.py` - Gymnasium wrapper.
- `python/train_ppo.py` - Stable-Baselines3 PPO trainer.
- `python/launch_envs.py` - launch multiple headless Godot env instances.
- `python/requirements.txt` - Python dependencies.

## Environment details

### Physics / stepping

- Fixed physics tick: `60 Hz` (`physics/common/physics_ticks_per_second=60`).
- Step action is held for `frame_skip` physics frames (default `3`).

### Action space (continuous, float32[4])

`[move_x, move_y, move_z, yaw_rate]`

- `move_x` / `move_z`: local horizontal desired velocity command
- `move_y`: local vertical thrust command
- `yaw_rate`: yaw angular command

All actions are expected in `[-1, 1]` and are clamped/normalized in the server.

### Observation (float32[8])

`[rel_target_x, rel_target_y, rel_target_z, local_vel_x, local_vel_y, local_vel_z, sin_yaw, cos_yaw]`

- Relative target position is in the agent's local frame.
- Velocity is agent local-frame velocity.

### Episode reset

- Supports optional deterministic seed (`int32`) in RESET request.
- Agent spawn and target position are randomized inside arena bounds.
- Agent velocity, yaw, timers, and step counters are reset.

### Reward / done

Per step reward:

- `progress = distance_prev - distance_now`
- `+success_bonus` when inside success radius
- `+time_penalty` each step
- `+collision_penalty` if collision during frame-skip window

Done conditions:

- success (inside success radius)
- timeout (`max_steps`)

## TCP protocol (single client)

Server: Godot instance (`scripts/Arena.gd`) listens on one TCP port.

Request messages are binary little-endian:

- **RESET**
  - `uint8 msg_type = 1`
  - `uint8 has_seed` (`0` or `1`)
  - optional `int32 seed` when `has_seed=1`
- **STEP**
  - `uint8 msg_type = 2`
  - `float32 action[4]`

Response (for both RESET and STEP):

- `uint32 obs_dim`
- `float32 obs[obs_dim]`
- `float32 reward`
- `uint8 done`

Robustness:

- validates message type and message sizes before parsing
- disconnects invalid clients cleanly
- handles disconnect/read/write errors

## Run one headless environment

From repo root:

```bash
/path/to/Godot_v4.2.2-stable_linux.x86_64 --headless --path . --scene res://scenes/Arena.tscn -- --port=9000
```

Important flags:

- `--headless` for no GPU window
- `--path .` project root
- `--scene ...` explicit scene entry
- `--` separates Godot args from user args (`--port`, etc.)

Optional user args (handled by `Arena.gd`):

- `--port=9000`
- `--frame-skip=3`
- `--max-steps=300`

## Python setup

```bash
cd python
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Quick client smoke test

```python
from env_client import RLGodotClient
import numpy as np

c = RLGodotClient(port=9000)
obs = c.reset(seed=123)
for _ in range(10):
    obs, rew, done = c.step(np.array([0.0, 0.0, 1.0, 0.0], dtype=np.float32))
    if done:
        obs = c.reset()
c.close()
```

## Launch multiple environments

```bash
python launch_envs.py --godot-bin /path/to/godot --project-path .. --start-port 9000 --num-envs 4
```

This launches ports `9000..9003`.

## Train PPO

In another shell (with envs already running):

```bash
cd python
python train_ppo.py --ports 9000 9001 9002 9003 --total-timesteps 200000
```

Default PPO hyperparameters (`python/train_ppo.py`):

- `learning_rate=3e-4`
- `n_steps=256`
- `batch_size=256`
- `gamma=0.99`
- `gae_lambda=0.95`
- `clip_range=0.2`
- `ent_coef=0.0`
- `vf_coef=0.5`

## Troubleshooting

- **Connection refused**: make sure Godot instance is running and port matches.
- **No response/hangs**: verify firewall/local security rules allow localhost TCP.
- **Wrong obs_dim**: check server and client protocol versions match.
- **Multiple env training fails**: ensure each Godot process has a unique port.
- **Headless launch issues**: verify the binary is Godot 4.2.x and executable.
