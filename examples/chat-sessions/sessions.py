"""Named local chat sessions with explicit, successful user/assistant turn pairs."""

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import sys
import tempfile
from typing import Any
import uuid

MAX_FILE_BYTES = 512_000
MAX_TURNS = 40
MAX_TEXT_CHARS = 4_000
SYSTEM_PROMPT = "You are a concise conversational assistant. Use the provided recent chat history. You have no tools."


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value):
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def exact_keys(value, keys, label):
    if not isinstance(value, dict) or set(value) != set(keys):
        raise ValueError(f"Invalid {label} fields")


def text_value(value, label, limit=MAX_TEXT_CHARS):
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{label} must contain 1 to {limit} characters and not be blank")


def session_path(store: Path, name: str) -> Path:
    if not isinstance(name, str) or re.fullmatch(r"[a-z][a-z0-9_-]{0,39}", name) is None:
        raise ValueError("Use a session name starting with a lowercase letter, followed by lowercase letters, digits, _ or -; at most 40 characters")
    if name in {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}:
        raise ValueError("Session name is a reserved Windows filename")
    return store / f"{name}.json"


def validate_state(state: Any, name: str) -> dict:
    exact_keys(state, {"schema_version", "session", "config", "turns"}, "session")
    if type(state["schema_version"]) is not int or state["schema_version"] != 1:
        raise ValueError("Only session schema_version=1 is supported")
    if state["session"] != name:
        raise ValueError("Session file belongs to a different name")
    config = state["config"]
    exact_keys(config, {"mode", "model_name", "system_prompt", "history_turns"}, "configuration")
    if config["mode"] not in ("scripted", "provider"):
        raise ValueError("Unknown session mode")
    text_value(config["model_name"], "model_name", 200)
    text_value(config["system_prompt"], "system_prompt")
    if type(config["history_turns"]) is not int or not 1 <= config["history_turns"] <= 10:
        raise ValueError("history_turns must be an integer from 1 to 10")
    if not isinstance(state["turns"], list) or len(state["turns"]) > MAX_TURNS:
        raise ValueError(f"Session may contain at most {MAX_TURNS} complete turns")
    for turn in state["turns"]:
        exact_keys(turn, {"user", "assistant"}, "turn")
        text_value(turn["user"], "user message")
        text_value(turn["assistant"], "assistant response")
    return state


def encode_state(state):
    raw = (json.dumps(state, indent=2, ensure_ascii=False) + "\n").encode("utf-8")
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Session exceeds its byte limit; start another named session")
    return raw


def load_session(store: Path, name: str) -> tuple[dict, bytes]:
    path = session_path(store, name)
    with path.open("rb") as stream:
        raw = stream.read(MAX_FILE_BYTES + 1)
    if len(raw) > MAX_FILE_BYTES:
        raise ValueError("Session file exceeds the byte limit")
    return validate_state(strict_json(raw.decode("utf-8")), name), raw


def create_session(store: Path, name: str, *, mode="scripted", model_name="gpt-4o-mini", history_turns=4):
    path = session_path(store, name)
    state = {"schema_version": 1, "session": name,
             "config": {"mode": mode, "model_name": model_name,
                        "system_prompt": SYSTEM_PROMPT, "history_turns": history_turns}, "turns": []}
    validate_state(state, name)
    raw = encode_state(state)
    store.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as stream:
        stream.write(raw)
    return state


def history_messages(state):
    selected = state["turns"][-state["config"]["history_turns"]:]
    return [{"role": role, "content": turn[key]} for turn in selected
            for role, key in (("user", "user"), ("assistant", "assistant"))]


class ScriptedModel:
    """Return supplied text; captures messages to test context handoff, not reasoning."""
    def __init__(self, answer):
        self.answer = answer
        self.calls = []

    def run(self, task=None, messages=None, **kwargs):
        self.calls.append(deepcopy(messages))
        return self.answer


def replace_session(path: Path, raw: bytes, previous: bytes):
    """Prepare a replacement before changing the existing session file.

    This is a single-writer example, not a cross-process locking protocol.
    """
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=f".{path.stem}-", suffix=".tmp", delete=False) as stream:
            temporary = Path(stream.name)
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        if path.read_bytes() != previous:
            raise ValueError("Session changed during this turn; reload before sending another message")
        os.replace(temporary, path)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def send_turn(store: Path, name: str, message: str, *, llm=None, workspace: Path) -> dict:
    text_value(message, "user message")
    state, previous = load_session(store, name)
    if len(state["turns"]) >= MAX_TURNS:
        raise ValueError(f"Session has reached {MAX_TURNS} turns; start a new session")
    mode = state["config"]["mode"]
    if (mode == "scripted") != (llm is not None):
        raise ValueError("Scripted sessions require a custom backend; provider sessions require llm=None")
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    from swarms import Agent

    agent = Agent(
        agent_name=f"chat-{name}-{uuid.uuid4().hex}", llm=llm,
        model_name=state["config"]["model_name"], system_prompt=state["config"]["system_prompt"],
        max_loops=1, retry_attempts=1, output_type="final",
        persistent_memory=False, autosave=False, context_compression=False,
        dynamic_context_window=False, streaming_on=False, stream=False, print_on=False,
    )
    messages = history_messages(state)
    start = len(agent.short_memory.conversation_history)
    answer = agent.run(task=message, messages=messages)
    if not any(row.get("role") == agent.agent_name for row in agent.short_memory.conversation_history[start:]):
        raise ValueError("Agent produced no model response; session was not updated")
    text_value(answer, "assistant response")
    updated = deepcopy(state)
    updated["turns"].append({"user": message, "assistant": answer})
    validate_state(updated, name)
    replace_session(session_path(store, name), encode_state(updated), previous)
    return {"session": name, "mode": mode, "turn": len(updated["turns"]),
            "history_turns_sent": len(messages) // 2,
            "older_turns_omitted": len(state["turns"]) - len(messages) // 2,
            "answer": answer}


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", type=Path, default=Path("chat-state"))
    commands = parser.add_subparsers(dest="command", required=True)
    create = commands.add_parser("create")
    create.add_argument("name")
    create.add_argument("--model", help="Opt into a provider-backed session; later sends may incur charges")
    create.add_argument("--history-turns", type=int, default=4)
    send = commands.add_parser("send")
    send.add_argument("name")
    send.add_argument("message")
    send.add_argument("--response", help="Required scripted response for a scripted session; omit for provider")
    send.add_argument("--workspace", type=Path, default=Path("agent_workspace_chat_sessions"))
    commands.add_parser("show").add_argument("name")
    args = parser.parse_args(argv)
    try:
        if args.command == "create":
            result = create_session(args.store, args.name,
                                    mode="provider" if args.model is not None else "scripted",
                                    model_name=args.model if args.model is not None else "gpt-4o-mini", history_turns=args.history_turns)
        elif args.command == "send":
            llm = ScriptedModel(args.response) if args.response is not None else None
            result = send_turn(args.store, args.name, args.message, llm=llm, workspace=args.workspace)
        else:
            result = load_session(args.store, args.name)[0]
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0
    except (OSError, ValueError) as exc:
        print(f"Chat session error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
