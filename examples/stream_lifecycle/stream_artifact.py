"""Preview one Swarms response; save only a verified, normally completed run."""

import argparse
import json
import math
import multiprocessing as mp
import os
import sys
import tempfile
import threading
import time
from contextlib import redirect_stdout
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Callable


CHUNKS = ("A draft", " can arrive", " in pieces.\n", "Complete: café 🌱\n")
SCENARIOS = {"success", "empty", "failure", "partial-failure", "stall"}


@dataclass(frozen=True)
class Outcome:
    status: str
    preview: str
    detail: str
    worker_pid: int
    worker_exitcode: int | None


class FixtureStream:
    """A scripted backend using the chunk shape consumed by real Swarms."""

    stream = False

    def __init__(self, scenario: str):
        self.scenario = scenario

    def run(self, task=None, messages=None, **kwargs):
        if self.scenario == "failure":
            raise RuntimeError("Scripted failure before any content")
        if self.scenario == "empty":
            return
        for index, text in enumerate(CHUNKS):
            yield SimpleNamespace(choices=[SimpleNamespace(
                delta=SimpleNamespace(content=text, tool_calls=None),
            )])
            if index == 0 and self.scenario == "partial-failure":
                raise RuntimeError("Scripted failure after preview content")
            if index == 0 and self.scenario == "stall":
                threading.Event().wait()  # Intentionally blocked; parent owns termination.


def _worker(connection, task: str, scenario: str, workspace: str, model: str | None):
    """Spawn target: construct the Agent here, never pickle a live Agent."""
    try:
        os.environ["WORKSPACE_DIR"] = workspace
        os.environ["SWARMS_TELEMETRY_ON"] = "false"
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        Path(workspace).mkdir(parents=True, exist_ok=True)
        with redirect_stdout(sys.stderr):
            from swarms import Agent
            from importlib.metadata import version

            agent = Agent(
                agent_name="DraftWriter",
                llm=None if model else FixtureStream(scenario),
                model_name=model or "gpt-4o-mini",
                system_prompt="Write a short plain-text draft for the supplied task.",
                max_loops=1, retry_attempts=1, output_type="final",
                streaming_on=True, stream=False, print_on=False,
                autosave=False, persistent_memory=False,
                context_compression=False, dynamic_context_window=False,
            )
            start = len(agent.short_memory.conversation_history)
            chunks = []

            def on_token(text):
                if not isinstance(text, str):
                    raise TypeError("Expected a text streaming callback")
                chunks.append(text)
                connection.send({"type": "token", "text": text})

            output = agent.run(task, streaming_callback=on_token)
            replies = [entry for entry in agent.short_memory.conversation_history[start:]
                       if entry.get("role") == agent.agent_name]
            # Exhausted LLM retries can be swallowed; final may then be the user task.
            if not replies:
                raise RuntimeError("No assistant reply was recorded")
            if not isinstance(output, str) or not output.strip():
                raise RuntimeError("The draft is empty or is not text")
            if output != "".join(chunks):
                raise RuntimeError("The final draft does not match the streamed preview")
            connection.send({"type": "complete", "output": output,
                             "framework_version": version("swarms"),
                             "mode": "provider" if model else "scripted"})
    except Exception as error:
        connection.send({"type": "error", "message": f"{type(error).__name__}: {error}"})
    finally:
        connection.close()


def _save_complete(path: Path, terminal: dict) -> None:
    """Replace the artifact only after a full, successful run has been accepted."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                         prefix=f".{path.name}.", suffix=".tmp", delete=False) as file:
            temporary = Path(file.name)
            json.dump({"status": "complete", **terminal}, file, ensure_ascii=False, indent=2)
            file.write("\n")
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary, path)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()


def run_to_artifact(task: str, output: Path, *, workspace: Path,
                    scenario: str = "success", timeout: float = 60.0,
                    cancel: threading.Event | None = None,
                    on_preview: Callable[[str], None] = lambda text: None,
                    model: str | None = None) -> Outcome:
    """Run one tool-free draft in a child; cancellation stops that local process.

    on_preview is synchronous and must return promptly. The timeout includes child
    startup and generation, but not the bounded process cleanup or artifact write.
    Passing model opts into a provider call and its credentials/costs.
    """
    if not isinstance(task, str) or not task.strip():
        raise ValueError("Supply a nonempty task")
    if scenario not in SCENARIOS:
        raise ValueError("Unknown fixture scenario")
    if isinstance(timeout, bool) or not isinstance(timeout, (int, float)) or not math.isfinite(timeout) or timeout <= 0:
        raise ValueError("Timeout must be a positive finite number")
    if model is not None and (not isinstance(model, str) or not model.strip()):
        raise ValueError("Model must be a nonempty provider identifier")
    context = mp.get_context("spawn")
    receiver, sender = context.Pipe(duplex=False)
    process = context.Process(target=_worker,
                              args=(sender, task, scenario, str(workspace.resolve()), model))
    preview = []
    terminal = None
    status, detail = "failed", "The child exited without a completion event"
    started = time.monotonic()
    process.start()
    sender.close()
    pid = process.pid
    try:
        while True:
            if cancel is not None and cancel.is_set():
                status, detail = "cancelled", "Local generation was cancelled"
                break
            if time.monotonic() - started >= timeout:
                status, detail = "timed_out", "Local generation exceeded its time limit"
                break
            if receiver.poll(0.05):
                try:
                    event = receiver.recv()
                except EOFError:
                    break
                if event["type"] == "token":
                    preview.append(event["text"])
                    on_preview(event["text"])
                elif event["type"] == "error":
                    detail = event["message"]
                    break
                elif event["type"] == "complete":
                    terminal = event
                    status, detail = "complete", "The child produced a complete draft"
                    break
            elif not process.is_alive():
                break
    except KeyboardInterrupt:
        status, detail = "cancelled", "Local generation was interrupted"
    finally:
        if status == "complete":
            process.join(2)
        if process.is_alive():
            process.terminate()
            process.join(2)
        if process.is_alive():
            process.kill()
            process.join(2)
        exitcode = process.exitcode
        receiver.close()
        process.close()
    if status == "complete" and exitcode != 0:
        status, detail = "failed", "The child did not exit normally after completion"
    if status == "complete":
        if cancel is not None and cancel.is_set():
            status, detail = "cancelled", "Cancellation was observed before artifact commit"
        elif terminal["output"] != "".join(preview):
            status, detail = "failed", "Completion does not match the received preview"
        else:
            _save_complete(output, terminal)
    return Outcome(status, "".join(preview), detail, pid, exitcode)


def main(argv=None):
    # Keep Unicode fixture text usable when Windows stdout is redirected.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("draft.json"))
    parser.add_argument("--workspace", type=Path, default=Path(".stream-workspace"))
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="success")
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--cancel-after-first", action="store_true")
    args = parser.parse_args(argv)
    cancel = threading.Event()

    def preview(text):
        print(text, end="", flush=True)
        if args.cancel_after_first:
            cancel.set()

    try:
        result = run_to_artifact("Explain why drafts need a completion state.", args.output,
                                 workspace=args.workspace, scenario=args.scenario,
                                 timeout=args.timeout, cancel=cancel, on_preview=preview)
    except (ValueError, OSError) as error:
        print(f"\nApplication error: {error}", file=sys.stderr)
        return 2
    print(f"\n{result.status}: {result.detail}")
    return 0 if result.status == "complete" else 1


if __name__ == "__main__":
    raise SystemExit(main())
