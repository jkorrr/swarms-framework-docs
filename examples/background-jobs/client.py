"""Submit once and poll a local agent job without resubmitting on a wait timeout."""

import argparse
import json
import math
from pathlib import Path
import sys
import time

import httpx


class PollingExpired(TimeoutError):
    pass


def poll(client, task_id, *, wait_seconds=30.0):
    if type(wait_seconds) not in (int, float) or not math.isfinite(wait_seconds) or wait_seconds < 0:
        raise ValueError("wait_seconds must be finite and nonnegative")
    deadline = time.monotonic() + wait_seconds
    first_request = True
    while True:
        if not first_request and time.monotonic() >= deadline:
            raise PollingExpired(f"Stopped waiting for {task_id}; the job was not cancelled. Poll this ID later.")
        first_request = False
        response = client.get(f"/jobs/{task_id}")
        response.raise_for_status()
        job = response.json()
        if job["status"] in {"succeeded", "failed", "cancelled"}:
            return job
        if job["status"] not in {"queued", "running"}:
            raise ValueError("Server returned an unknown job status")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise PollingExpired(f"Stopped waiting for {task_id}; the job was not cancelled. Poll this ID later.")
        time.sleep(min(0.2, remaining))


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8030")
    parser.add_argument("--wait", type=float, default=30.0)
    parser.add_argument("--output", type=Path)
    actions = parser.add_subparsers(dest="action", required=True)
    actions.add_parser("submit").add_argument("task")
    actions.add_parser("poll").add_argument("task_id")
    actions.add_parser("cancel").add_argument("task_id")
    args = parser.parse_args()
    try:
        if not math.isfinite(args.wait) or args.wait < 0:
            raise ValueError("--wait must be finite and nonnegative")
        with httpx.Client(base_url=args.url, timeout=5.0, trust_env=False) as client:
            if args.action == "submit":
                response = client.post("/jobs", json={"task": args.task})
                response.raise_for_status()
                task_id = response.json()["id"]
                print(f"Accepted job: {task_id}", flush=True)
                job = poll(client, task_id, wait_seconds=args.wait)
            elif args.action == "poll":
                job = poll(client, args.task_id, wait_seconds=args.wait)
            else:
                response = client.delete(f"/jobs/{args.task_id}")
                response.raise_for_status()
                job = response.json()
        text = json.dumps(job, indent=2, ensure_ascii=False)
        if args.output:
            args.output.write_text(text + "\n", encoding="utf-8")
        print(text)
        return 0 if job["status"] == "succeeded" or args.action == "cancel" else 1
    except (httpx.HTTPError, OSError, ValueError) as exc:
        print(f"Client stopped: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
