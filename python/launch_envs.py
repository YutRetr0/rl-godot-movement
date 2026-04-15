from __future__ import annotations

import argparse
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Launch multiple headless Godot RL env instances")
    parser.add_argument(
        "--godot-bin",
        type=str,
        required=True,
        help="Path to Godot 4.x executable",
    )
    parser.add_argument(
        "--project-path",
        type=str,
        default=str(Path(__file__).resolve().parents[1]),
        help="Path to Godot project root",
    )
    parser.add_argument("--start-port", type=int, default=9000)
    parser.add_argument("--num-envs", type=int, default=4)
    parser.add_argument("--frame-skip", type=int, default=3)
    parser.add_argument("--max-steps", type=int, default=300)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    processes: list[subprocess.Popen] = []

    print(f"Launching {args.num_envs} envs starting at port {args.start_port}")
    for i in range(args.num_envs):
        port = args.start_port + i
        cmd = [
            args.godot_bin,
            "--headless",
            "--path",
            args.project_path,
            "--scene",
            "res://scenes/Arena.tscn",
            "--",
            f"--port={port}",
            f"--frame-skip={args.frame_skip}",
            f"--max-steps={args.max_steps}",
        ]
        proc = subprocess.Popen(cmd)
        processes.append(proc)
        print(f"Started env on port {port} with PID {proc.pid}")

    def _shutdown(_signum, _frame):
        for proc in processes:
            if proc.poll() is None:
                proc.terminate()
        for proc in processes:
            if proc.poll() is None:
                proc.wait(timeout=10)
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    try:
        while True:
            alive = [proc for proc in processes if proc.poll() is None]
            if not alive:
                break
            time.sleep(1.0)
    finally:
        _shutdown(None, None)


if __name__ == "__main__":
    main()
