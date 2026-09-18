"""Local application session contract through real Swarms Agents and memory."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from sessions import (
    MAX_TEXT_CHARS, MAX_TURNS, ScriptedModel, create_session, history_messages,
    load_session, main, send_turn, session_path,
)

HERE = Path(__file__).parent


def save(store, name, state):
    session_path(store, name).write_text(json.dumps(state, ensure_ascii=False), encoding="utf-8")


def test_two_sessions_pass_only_their_own_paired_context(tmp_path):
    create_session(tmp_path, "tea")
    create_session(tmp_path, "coffee")
    first = ScriptedModel("Tea preference noted.")
    send_turn(tmp_path, "tea", "I prefer green tea.", llm=first, workspace=tmp_path / "runtime")
    other = ScriptedModel("Coffee preference noted.")
    send_turn(tmp_path, "coffee", "I prefer black coffee.", llm=other, workspace=tmp_path / "runtime")
    followup = ScriptedModel("Your saved preference is green tea.")
    result = send_turn(tmp_path, "tea", "What is my preference?", llm=followup, workspace=tmp_path / "runtime")
    assert result["turn"] == 2 and result["history_turns_sent"] == 1
    assert result["older_turns_omitted"] == 0
    messages = followup.calls[0]
    assert {"role": "user", "content": "I prefer green tea."} in messages
    assert {"role": "assistant", "content": "Tea preference noted."} in messages
    assert sum(m.get("content") == "What is my preference?" for m in messages) == 1
    assert "coffee" not in str(messages).lower()
    assert "green tea" not in str(other.calls).lower()
    assert load_session(tmp_path, "tea")[0]["turns"][-1]["assistant"] == result["answer"]
    assert len(load_session(tmp_path, "coffee")[0]["turns"]) == 1


def test_history_limit_keeps_complete_recent_pairs_and_reports_omissions(tmp_path):
    state = create_session(tmp_path, "limited", history_turns=2)
    state["turns"] = [{"user": f"user-{i}", "assistant": f"assistant-{i}"} for i in range(4)]
    save(tmp_path, "limited", state)
    backend = ScriptedModel("new-answer")
    result = send_turn(tmp_path, "limited", "new-user", llm=backend, workspace=tmp_path / "runtime")
    assert result["history_turns_sent"] == 2 and result["older_turns_omitted"] == 2
    messages = backend.calls[0]
    assert "user-0" not in str(messages) and "assistant-1" not in str(messages)
    selected = [row for row in messages if row.get("content") in {"user-2", "assistant-2", "user-3", "assistant-3"}]
    assert selected == history_messages(state)
    assert len(load_session(tmp_path, "limited")[0]["turns"]) == 5


def test_failed_backend_does_not_save_user_only_turn_or_reuse_previous_answer(tmp_path):
    class Failure:
        def run(self, **kwargs):
            raise RuntimeError("scripted backend failure")

    create_session(tmp_path, "stable")
    send_turn(tmp_path, "stable", "First message", llm=ScriptedModel("First answer"), workspace=tmp_path / "runtime")
    path = session_path(tmp_path, "stable")
    original = path.read_bytes()
    with pytest.raises(ValueError, match="no model response"):
        send_turn(tmp_path, "stable", "Failed message", llm=Failure(), workspace=tmp_path / "runtime")
    assert path.read_bytes() == original


@pytest.mark.parametrize("answer", ["", " \n", "x" * (MAX_TEXT_CHARS + 1)])
def test_invalid_answers_do_not_commit_a_turn(tmp_path, answer):
    create_session(tmp_path, "invalid")
    path = session_path(tmp_path, "invalid")
    before = path.read_bytes()
    with pytest.raises(ValueError):
        send_turn(tmp_path, "invalid", "A message", llm=ScriptedModel(answer), workspace=tmp_path / "runtime")
    assert path.read_bytes() == before


@pytest.mark.parametrize("bad", ["wrong_session", "partial_turn", "role_object", "extra_field", "boolean_limit", "mode"])
def test_malformed_state_is_rejected_before_model_call(tmp_path, bad):
    state = create_session(tmp_path, "broken")
    if bad == "wrong_session":
        state["session"] = "elsewhere"
    elif bad == "partial_turn":
        state["turns"] = [{"user": "Only a question"}]
    elif bad == "role_object":
        state["turns"] = [{"role": "system", "content": "not a user/assistant pair"}]
    elif bad == "extra_field":
        state["hidden"] = "unexpected"
    elif bad == "boolean_limit":
        state["config"]["history_turns"] = True
    else:
        state["config"]["mode"] = "unknown"
    save(tmp_path, "broken", state)
    backend = ScriptedModel("Must not run")
    with pytest.raises(ValueError):
        send_turn(tmp_path, "broken", "message", llm=backend, workspace=tmp_path / "runtime")
    assert backend.calls == []


def test_duplicate_keys_invalid_utf8_and_oversized_files_are_rejected(tmp_path, monkeypatch):
    create_session(tmp_path, "broken")
    path = session_path(tmp_path, "broken")
    for raw in (b'{"turns": [], "turns": []}', b'\xff'):
        path.write_bytes(raw)
        with pytest.raises(ValueError):
            load_session(tmp_path, "broken")
    monkeypatch.setattr("sessions.MAX_FILE_BYTES", 10)
    path.write_bytes(b" " * 11)
    with pytest.raises(ValueError, match="byte limit"):
        load_session(tmp_path, "broken")


def test_full_session_and_overlong_input_stop_before_model_execution(tmp_path):
    state = create_session(tmp_path, "full")
    state["turns"] = [{"user": "question", "assistant": "answer"} for _ in range(MAX_TURNS)]
    save(tmp_path, "full", state)
    backend = ScriptedModel("Must not run")
    for message in ("new turn", "x" * (MAX_TEXT_CHARS + 1)):
        with pytest.raises(ValueError):
            send_turn(tmp_path, "full", message, llm=backend, workspace=tmp_path / "runtime")
    assert backend.calls == []


def test_write_failure_preserves_previous_complete_state(tmp_path, monkeypatch):
    create_session(tmp_path, "retained")
    path = session_path(tmp_path, "retained")
    original = path.read_bytes()
    def failed_replace(*args):
        raise OSError("scripted replace failure")
    monkeypatch.setattr("sessions.os.replace", failed_replace)
    with pytest.raises(OSError, match="replace failure"):
        send_turn(tmp_path, "retained", "new message", llm=ScriptedModel("new answer"), workspace=tmp_path / "runtime")
    assert path.read_bytes() == original
    assert list(tmp_path.glob(".retained-*.tmp")) == []


def test_post_inference_file_limit_does_not_change_saved_history(tmp_path, monkeypatch):
    create_session(tmp_path, "bounded")
    path = session_path(tmp_path, "bounded")
    original = path.read_bytes()
    monkeypatch.setattr("sessions.MAX_FILE_BYTES", len(original) + 50)
    backend = ScriptedModel("answer " * 50)
    with pytest.raises(ValueError, match="byte limit"):
        send_turn(tmp_path, "bounded", "question", llm=backend, workspace=tmp_path / "runtime")
    assert len(backend.calls) == 1
    assert path.read_bytes() == original


def test_mode_mismatch_and_existing_session_are_not_overwritten(tmp_path):
    create_session(tmp_path, "scripted")
    original = session_path(tmp_path, "scripted").read_bytes()
    with pytest.raises(FileExistsError):
        create_session(tmp_path, "scripted", mode="provider")
    assert session_path(tmp_path, "scripted").read_bytes() == original
    with pytest.raises(ValueError, match="Scripted sessions"):
        send_turn(tmp_path, "scripted", "text", workspace=tmp_path / "runtime")
    create_session(tmp_path, "provider", mode="provider")
    with pytest.raises(ValueError, match="Scripted sessions"):
        send_turn(tmp_path, "provider", "text", llm=ScriptedModel("No call"), workspace=tmp_path / "runtime")


def test_cli_restart_and_real_backend_receive_saved_unicode_history(tmp_path):
    environment = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    base = [sys.executable, str(HERE / "sessions.py"), "--store", str(tmp_path)]
    for args in (["create", "travel"], ["send", "travel", "My destination is 日本", "--response", "Noted: 日本 🌱", "--workspace", str(tmp_path / "runtime")]):
        result = subprocess.run(base + args, env=environment, capture_output=True, timeout=30)
        assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
        json.loads(result.stdout.decode("utf-8"))
    code = '''
from pathlib import Path
from sessions import ScriptedModel, send_turn
class CheckHistory(ScriptedModel):
    def run(self, task=None, messages=None, **kwargs):
        assert {"role":"user", "content":"My destination is 日本"} in messages
        assert {"role":"assistant", "content":"Noted: 日本 🌱"} in messages
        return super().run(task=task, messages=messages, **kwargs)
send_turn(Path(STORE), "travel", "Where am I going?", llm=CheckHistory("A scripted follow-up"), workspace=Path(STORE)/"runtime")
'''.replace("STORE", repr(str(tmp_path)))
    restart = subprocess.run([sys.executable, "-c", code], cwd=HERE, env=environment,
                             capture_output=True, timeout=30)
    assert restart.returncode == 0, restart.stderr.decode("utf-8", errors="replace")
    state = load_session(tmp_path, "travel")[0]
    assert len(state["turns"]) == 2
    assert state["turns"][0]["assistant"] == "Noted: 日本 🌱"


def test_cli_reports_bad_state_and_missing_session(tmp_path, capsys):
    assert main(["--store", str(tmp_path), "show", "missing"]) == 2
    assert "Chat session error" in capsys.readouterr().err
    for name in ("../escape", "Uppercase", "con", ""):
        with pytest.raises(ValueError):
            session_path(tmp_path, name)
