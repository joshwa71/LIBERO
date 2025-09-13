#!/usr/bin/env python3

import argparse
import base64
import json
import os
import socket
import struct
import sys
import threading
from typing import Any, Dict

import numpy as np

from libero.libero import benchmark, get_libero_path
from libero.libero.envs.env_wrapper import OffScreenRenderEnv, OnScreenRenderEnv


def _encode_ndarray(arr: np.ndarray) -> Dict[str, Any]:
    return {
        "__ndarray__": True,
        "dtype": str(arr.dtype),
        "shape": list(arr.shape),
        "data": base64.b64encode(arr.tobytes(order="C")).decode("ascii"),
    }


def _decode_ndarray(obj: Dict[str, Any]) -> np.ndarray:
    assert obj.get("__ndarray__", False)
    dtype = np.dtype(obj["dtype"])  # type: ignore[arg-type]
    shape = tuple(obj["shape"])  # type: ignore[assignment]
    raw = base64.b64decode(obj["data"])  # type: ignore[arg-type]
    arr = np.frombuffer(raw, dtype=dtype)
    return arr.reshape(shape)


def _send_msg(conn: socket.socket, msg: Dict[str, Any]) -> None:
    data = json.dumps(msg).encode("utf-8")
    header = struct.pack(">I", len(data))
    conn.sendall(header + data)


def _recv_exact(conn: socket.socket, n: int) -> bytes:
    buf = bytearray()
    while len(buf) < n:
        chunk = conn.recv(n - len(buf))
        if not chunk:
            raise ConnectionError("client closed")
        buf.extend(chunk)
    return bytes(buf)


def _recv_msg(conn: socket.socket) -> Dict[str, Any]:
    header = _recv_exact(conn, 4)
    (size,) = struct.unpack(">I", header)
    payload = _recv_exact(conn, size)
    return json.loads(payload.decode("utf-8"))


class LiberoRemoteServer:
    def __init__(self, benchmark_name: str, task_id: int, host: str, port: int, height: int, width: int, on_screen: bool):
        self.benchmark_name = benchmark_name
        self.task_id = task_id
        self.host = host
        self.port = port
        self.height = height
        self.width = width

        self.on_screen = on_screen
        self.env = self._make_env()

    def _make_env(self) -> OffScreenRenderEnv:
        bench = benchmark.get_benchmark_dict()[self.benchmark_name]()
        task = bench.get_task(self.task_id)
        bddl_path = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)

        EnvClass = OnScreenRenderEnv if self.on_screen else OffScreenRenderEnv
        env = EnvClass(
            bddl_file_name=bddl_path,
            camera_heights=self.height,
            camera_widths=self.width,
        )
        env.reset()
        return env

    def _obs_to_message(self, obs: Dict[str, Any]) -> Dict[str, Any]:
        # Expected keys from LIBERO env: agentview_image, robot0_eye_in_hand_image, robot0_eef_pos, etc.
        pixels = {}
        if "agentview_image" in obs:
            # Use 'head' to match trained policy configs
            pixels["head"] = _encode_ndarray(obs["agentview_image"].astype(np.uint8, copy=False))
        if "robot0_eye_in_hand_image" in obs:
            pixels["wrist"] = _encode_ndarray(obs["robot0_eye_in_hand_image"].astype(np.uint8, copy=False))

        # Build a compact state in the order used during training:
        # [ee_pos(3), ee_ori_axis_angle(3), gripper(2), joints(7)] => 15 dims
        state_vecs = []
        # 1) End-effector position
        if "robot0_eef_pos" in obs:
            state_vecs.append(obs["robot0_eef_pos"].astype(np.float32, copy=False).reshape(-1))
        else:
            state_vecs.append(np.zeros((3,), dtype=np.float32))
        # 2) End-effector orientation (axis-angle)
        if "robot0_eef_quat" in obs:
            q = obs["robot0_eef_quat"].astype(np.float32, copy=False).reshape(-1)
            # Robustly convert quaternion to axis-angle.
            # Robosuite typically uses XYZW ordering; assume (x, y, z, w) and normalize.
            if q.shape[0] == 4:
                x, y, z, w = q[0], q[1], q[2], q[3]
                norm = max(1e-8, float(np.sqrt(w * w + x * x + y * y + z * z)))
                w, x, y, z = w / norm, x / norm, y / norm, z / norm
                angle = 2.0 * float(np.arccos(np.clip(w, -1.0, 1.0)))
                s = float(np.sqrt(max(1e-8, 1.0 - w * w)))
                if s > 1e-6:
                    axis = np.array([x / s, y / s, z / s], dtype=np.float32)
                else:
                    axis = np.array([0.0, 0.0, 0.0], dtype=np.float32)
                axis_angle = axis * angle
                state_vecs.append(axis_angle.astype(np.float32, copy=False).reshape(-1))
            else:
                state_vecs.append(np.zeros((3,), dtype=np.float32))
        else:
            state_vecs.append(np.zeros((3,), dtype=np.float32))
        # 3) Gripper
        if "robot0_gripper_qpos" in obs:
            state_vecs.append(obs["robot0_gripper_qpos"].astype(np.float32, copy=False).reshape(-1))
        else:
            state_vecs.append(np.zeros((2,), dtype=np.float32))
        # 4) Joints
        if "robot0_joint_pos" in obs:
            state_vecs.append(obs["robot0_joint_pos"].astype(np.float32, copy=False).reshape(-1))
        else:
            state_vecs.append(np.zeros((7,), dtype=np.float32))

        # Compose and force exactly 15 dims
        state = np.concatenate(state_vecs, axis=0) if state_vecs else np.zeros((15,), dtype=np.float32)
        if state.size < 15:
            state = np.concatenate([state, np.zeros((15 - state.size,), dtype=np.float32)], axis=0)
        elif state.size > 15:
            state = state[:15]

        return {"pixels": pixels, "agent_pos": _encode_ndarray(state)}

    def _spec_message(self) -> Dict[str, Any]:
        # Advertise 7-dim OSC action; server also accepts a 4-dim shorthand at step()
        # Bounds are abstract; client re-normalizes for policy.
        image_shapes = {"head": [self.height, self.width, 3], "wrist": [self.height, self.width, 3]}

        # Compute state dimension from a fresh observation
        try:
            obs = self.env.reset()
        except Exception:
            obs = {}
        # reuse concatenation logic
        state_vecs = []
        if isinstance(obs, dict):
            if "robot0_joint_pos" in obs:
                state_vecs.append(obs["robot0_joint_pos"].astype(np.float32, copy=False).reshape(-1))
            if "robot0_gripper_qpos" in obs:
                state_vecs.append(obs["robot0_gripper_qpos"].astype(np.float32, copy=False).reshape(-1))
            if "robot0_eef_pos" in obs:
                state_vecs.append(obs["robot0_eef_pos"].astype(np.float32, copy=False).reshape(-1))
        # Mirror the 15-dim state guarantee from _obs_to_message
        state = np.concatenate(state_vecs, axis=0) if state_vecs else np.zeros((15,), dtype=np.float32)
        if state.size < 15:
            state = np.concatenate([state, np.zeros((15 - state.size,), dtype=np.float32)], axis=0)
        elif state.size > 15:
            state = state[:15]
        state_dim = 15

        return {
            "status": "ok",
            "action_dim": 7,
            "action_low": -1.0,
            "action_high": 1.0,
            "state_dim": state_dim,
            "image_shapes": image_shapes,
        }

    def serve_client(self, conn: socket.socket) -> None:
        try:
            while True:
                msg = _recv_msg(conn)
                cmd = msg.get("cmd")
                if cmd == "hello":
                    _send_msg(conn, self._spec_message())
                elif cmd == "reset":
                    obs = self.env.reset()
                    _send_msg(conn, {"status": "ok", "observation": self._obs_to_message(obs), "info": {}})
                elif cmd == "step":
                    act = _decode_ndarray(msg["action"])  # type: ignore[index]
                    # Support two input formats:
                    # - 7-dim OSC pose: [dx, dy, dz, dRx, dRy, dRz, gripper]
                    # - 4-dim shorthand: [dx, dy, dz, gripper] → expand with zeros for rotation
                    if act.shape[0] == 7:
                        action_rs = act.astype(np.float32, copy=False)
                    elif act.shape[0] >= 4:
                        action_rs = np.zeros((7,), dtype=np.float32)
                        action_rs[:3] = act[:3]
                        g = float(act[3])
                        # Symmetric thresholding for gripper in [-1, 1]
                        action_rs[-1] = -1.0 if g < -0.5 else (1.0 if g > 0.5 else 0.0)
                    else:
                        action_rs = np.zeros((7,), dtype=np.float32)

                    obs, reward, done, info = self.env.step(action_rs)
                    # Ensure on-screen viewer updates
                    if self.on_screen and hasattr(self.env, "render"):
                        try:
                            self.env.render()
                        except Exception:
                            pass
                    terminated = bool(done)
                    truncated = False
                    _send_msg(
                        conn,
                        {
                            "status": "ok",
                            "observation": self._obs_to_message(obs),
                            "reward": float(reward),
                            "terminated": terminated,
                            "truncated": truncated,
                            "info": info or {},
                        },
                    )
                elif cmd == "close":
                    _send_msg(conn, {"status": "ok"})
                    break
                else:
                    _send_msg(conn, {"status": "error", "message": f"unknown cmd {cmd}"})
        finally:
            try:
                conn.close()
            except Exception:
                pass

    def run(self) -> None:
        srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind((self.host, self.port))
        srv.listen(1)
        print(f"[libero-remote] listening on {self.host}:{self.port}")
        try:
            while True:
                conn, addr = srv.accept()
                print(f"[libero-remote] client connected: {addr}")
                t = threading.Thread(target=self.serve_client, args=(conn,), daemon=True)
                t.start()
        finally:
            try:
                srv.close()
            except Exception:
                pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", type=str, default="libero_10")
    parser.add_argument("--task_id", type=int, default=0)
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--height", type=int, default=128)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--onscreen", action="store_true", help="Enable on-screen Mujoco viewer")
    args = parser.parse_args()

    try:
        srv = LiberoRemoteServer(
            benchmark_name=args.benchmark,
            task_id=args.task_id,
            host=args.host,
            port=args.port,
            height=args.height,
            width=args.width,
            on_screen=args.onscreen,
        )
        srv.run()
    except KeyboardInterrupt:
        print("[libero-remote] shutting down")
        sys.exit(0)


if __name__ == "__main__":
    main()


