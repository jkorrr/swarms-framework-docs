"""Exercise the real Swarms streaming path in real spawned child processes."""

import json
import multiprocessing as mp
import os
from pathlib import Path
import subprocess
import sys
import threading

import pytest

from stream_artifact import CHUNKS, run_to_artifact


def track_processes(monkeypatch):
    """Observe public closed-handle behavior without replacing real execution."""
    context = mp.get_context("spawn")
    processes = []

    class ObservedContext:
        Pipe = staticmethod(context.Pipe)

        def Process(self, *args, **kwargs):
            process = context.Process(*args, **kwargs)
            processes.append(process)
            return process

    monkeypatch.setattr(mp, "get_context", lambda method: ObservedContext())
    return processes


def test_success_streams_before_completion_and_commits_unicode(tmp_path):
    path = tmp_path / "draft.json"
    path.write_text("previous accepted artifact", encoding="utf-8")
    chunks = []

    def preview(text):
        assert path.read_text(encoding="utf-8") == "previous accepted artifact"
        chunks.append(text)

    result = run_to_artifact("Write a draft", path, workspace=tmp_path / "runtime", on_preview=preview)
    assert chunks == list(CHUNKS)
    assert result.status == "complete"
    assert result.worker_exitcode == 0
    artifact = json.loads(path.read_text(encoding="utf-8"))
    assert artifact["status"] == "complete"
    assert artifact["mode"] == "scripted"
    assert artifact["framework_version"] == "15.0.3"
    assert artifact["output"] == result.preview == "".join(CHUNKS)
    assert list(tmp_path.glob(".draft.json.*.tmp")) == []


@pytest.mark.parametrize("scenario,expected_preview", [
    ("failure", ""), ("empty", ""), ("partial-failure", CHUNKS[0]),
])
def test_failed_real_agent_does_not_replace_existing_artifact(tmp_path, scenario, expected_preview):
    path = tmp_path / "draft.json"
    path.write_text("previous accepted artifact", encoding="utf-8")
    # A swallowed backend failure must not save the input as the final answer.
    result = run_to_artifact("An apparently complete draft.", path,
                             workspace=tmp_path / "runtime", scenario=scenario)
    assert result.status == "failed"
    assert result.preview == expected_preview
    assert result.worker_exitcode is not None
    assert path.read_text(encoding="utf-8") == "previous accepted artifact"


def test_cancel_after_first_real_token_terminates_blocked_child(tmp_path):
    path = tmp_path / "draft.json"
    cancel = threading.Event()
    chunks = []

    def preview(text):
        chunks.append(text)
        cancel.set()

    result = run_to_artifact("Write a draft", path, workspace=tmp_path / "runtime",
                             scenario="stall", cancel=cancel, on_preview=preview)
    assert chunks == [CHUNKS[0]]
    assert result.status == "cancelled"
    assert result.worker_exitcode not in (None, 0)
    assert all(child.pid != result.worker_pid for child in mp.active_children())
    assert not path.exists()


def test_timeout_stops_a_real_child_and_preserves_partial_preview(tmp_path):
    path = tmp_path / "draft.json"
    # Startup is intentionally part of the deadline. On a slow host this can
    # expire before the first token; either way no completed artifact is valid.
    result = run_to_artifact("Write a draft", path, workspace=tmp_path / "runtime",
                             scenario="stall", timeout=10)
    assert result.status == "timed_out"
    assert result.preview in ("", CHUNKS[0])
    assert result.worker_exitcode not in (None, 0)
    assert all(child.pid != result.worker_pid for child in mp.active_children())
    assert not path.exists()


def test_preview_sink_failure_also_cleans_up_child(tmp_path, monkeypatch):
    def broken_sink(text):
        raise LookupError("Scripted consumer failure")

    before = {child.pid for child in mp.active_children()}
    processes = track_processes(monkeypatch)
    with pytest.raises(LookupError, match="consumer failure"):
        run_to_artifact("Write a draft", tmp_path / "draft.json", workspace=tmp_path / "runtime",
                        scenario="stall", on_preview=broken_sink)
    assert {child.pid for child in mp.active_children()} == before
    assert not (tmp_path / "draft.json").exists()
    with pytest.raises(ValueError, match="closed"):
        processes[0].is_alive()


def test_artifact_write_failure_closes_child_handle_and_removes_temporary(tmp_path, monkeypatch):
    destination = tmp_path / "existing-directory"
    destination.mkdir()
    processes = track_processes(monkeypatch)
    with pytest.raises(OSError):
        run_to_artifact("Write a draft", destination, workspace=tmp_path / "runtime")
    assert destination.is_dir()
    assert list(tmp_path.glob(".existing-directory.*.tmp")) == []
    with pytest.raises(ValueError, match="closed"):
        processes[0].is_alive()


@pytest.mark.parametrize("timeout", [0, -1, float("nan"), float("inf"), True])
def test_invalid_timeout_is_rejected_before_starting_a_child(tmp_path, timeout):
    with pytest.raises(ValueError, match="Timeout"):
        run_to_artifact("Write a draft", tmp_path / "draft.json", workspace=tmp_path,
                        timeout=timeout)


@pytest.mark.parametrize("arguments,status,exitcode", [
    ([], "complete", 0),
    (["--scenario", "partial-failure"], "failed", 1),
    (["--scenario", "stall", "--cancel-after-first"], "cancelled", 1),
])
def test_documented_cli_with_redirected_windows_style_stdout(tmp_path, arguments, status, exitcode):
    path = tmp_path / "draft.json"
    path.write_text("previous accepted artifact", encoding="utf-8")
    environment = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    result = subprocess.run(
        [sys.executable, str(Path(__file__).with_name("stream_artifact.py")),
         "--output", str(path), "--workspace", str(tmp_path / "runtime"), *arguments],
        cwd=tmp_path, env=environment, capture_output=True, timeout=60,
    )
    assert result.returncode == exitcode, result.stderr.decode("cp1252", errors="replace")
    assert f"{status}:" in result.stdout.decode("utf-8")
    if status == "complete":
        assert "café 🌱" in result.stdout.decode("utf-8")
        assert json.loads(path.read_text(encoding="utf-8"))["output"] == "".join(CHUNKS)
    else:
        assert path.read_text(encoding="utf-8") == "previous accepted artifact"
