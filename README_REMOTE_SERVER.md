# LIBERO Remote Environment Server

Run LIBERO tasks as a remote gym-like server so external clients (e.g., `lerobot-upstream`) can connect without sharing dependencies.

## Start the server
```bash
cd /home/josh/phddev/LIBERO
python scripts/remote_env_server.py \
  --benchmark libero_10 \
  --task_id 0 \
  --host 127.0.0.1 \
  --port 5555 \
  --height 128 \
  --width 128
```

- Uses `OffScreenRenderEnv` under the hood.
- Streams observations: two RGB cameras (`agentview`, `wrist`) and a compact state vector.
- Accepts a 4D action `[dx, dy, dz, gripper]` and maps it to robosuite OSC pose.

## Protocol (for reference)
- TCP framing: big-endian 4-byte length header + JSON payload.
- Messages:
  - Client → Server:
    - `{ "cmd": "hello" }`
    - `{ "cmd": "reset" }`
    - `{ "cmd": "step", "action": encoded_ndarray }`
    - `{ "cmd": "close" }`
  - Server → Client:
    - On `hello`: spec with `action_dim`, `image_shapes`, `state_dim`.
    - On `reset` / `step`: `{ "status": "ok", "observation": { ... }, ... }`.

All tensors are encoded with base64 in JSON. See `scripts/remote_env_server.py` for details.
