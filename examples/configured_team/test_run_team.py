"""Check preflight and exact configured handoffs using real released Swarms objects."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest
import yaml

import run_team
from run_team import AGENT_POLICY, HERE, fingerprint, main, read_config, replay_config, run_config


@pytest.fixture
def config():
    return read_config((HERE / "team.yaml").read_text(encoding="utf-8"))


@pytest.fixture
def responses():
    return json.loads((HERE / "responses.json").read_text(encoding="utf-8"))


def test_real_loader_agent_and_workflow_handoffs(config, responses, tmp_path, monkeypatch):
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from swarms import Agent
    from swarms.utils.litellm_wrapper import LiteLLM

    monkeypatch.setattr(LiteLLM, "run", lambda *args, **kwargs: pytest.fail("Provider inference is forbidden in fixture mode"))
    captured = []
    original = run_team.build_team

    def observed(*args, **kwargs):
        agents = original(*args, **kwargs)
        captured.extend(agents)
        return agents

    monkeypatch.setattr(run_team, "build_team", observed)
    report = run_config(config, responses=responses, workspace=tmp_path)
    assert report["status"] == "complete"
    assert report["final_output"] == responses["Writer"]
    assert report["mode"] == "scripted"
    assert report["config"] == config.model_dump()
    assert [stage["agent_name"] for stage in report["stages"]] == ["Outline", "Writer"]
    assert all(isinstance(agent, Agent) for agent in captured)
    assert captured[0] is not captured[1]
    for agent, definition in zip(captured, config.agents):
        assert agent.system_prompt == definition.system_prompt
        assert agent.temperature == definition.temperature
        assert all(getattr(agent, key) == value for key, value in AGENT_POLICY.items())
        assert len(agent.llm.messages) == 1
    assert any(message.get("content") == config.task for message in captured[0].llm.messages[0])
    assert responses["Outline"] in str(captured[1].llm.messages[0])


@pytest.mark.parametrize("replacement", [
    {"schema_version": True}, {"schema_version": 2}, {"task": " "},
    {"agents": []}, {"unknown_option": True},
])
def test_invalid_top_level_config_is_rejected(config, replacement):
    data = {**config.model_dump(), **replacement}
    with pytest.raises(ValueError):
        read_config(yaml.safe_dump(data))


@pytest.mark.parametrize("mutation", ["duplicate_name", "typo", "flow_syntax", "blank_prompt", "nan_temperature"])
def test_invalid_agent_config_is_rejected(config, mutation):
    data = config.model_dump()
    if mutation == "duplicate_name":
        data["agents"][1]["agent_name"] = data["agents"][0]["agent_name"]
    elif mutation == "typo":
        data["agents"][0]["temprature"] = 0.2
    elif mutation == "flow_syntax":
        data["agents"][0]["agent_name"] = "Outline -> Writer"
    elif mutation == "blank_prompt":
        data["agents"][0]["system_prompt"] = "  "
    else:
        data["agents"][0]["temperature"] = float("nan")
    with pytest.raises(ValueError):
        read_config(yaml.safe_dump(data))


def test_duplicate_yaml_keys_are_not_silently_overwritten():
    with pytest.raises(ValueError, match="Duplicate YAML key.*line 2"):
        read_config("schema_version: 1\nschema_version: 2\n")


def test_validate_cli_does_not_import_swarms_or_create_workspace(tmp_path):
    code = (
        "import sys; from run_team import main; "
        "assert main(['validate', sys.argv[1]]) == 0; "
        "assert 'swarms' not in sys.modules"
    )
    result = subprocess.run([sys.executable, "-c", code, str(HERE / "team.yaml")], cwd=tmp_path,
                            env={**os.environ, "PYTHONPATH": str(HERE)}, capture_output=True, timeout=15)
    assert result.returncode == 0, result.stderr.decode("utf-8", errors="replace")
    assert not (tmp_path / "agent_workspace_configured").exists()


def test_bad_config_cli_reports_location_before_any_build(tmp_path, monkeypatch):
    path = tmp_path / "wrong.yaml"
    path.write_text((HERE / "team.yaml").read_text().replace("temperature:", "temprature:"), encoding="utf-8")
    monkeypatch.setattr(run_team, "build_team", lambda *args, **kwargs: pytest.fail("Invalid config reached Agent construction"))
    assert main(["run", str(path), "--output", str(tmp_path / "result.json")]) == 2
    assert not (tmp_path / "result.json").exists()


def test_invalid_unicode_config_cli_has_readable_stderr(tmp_path):
    source = tmp_path / "invalid-🌱.yaml"
    source.write_text("🌱: first\n🌱: second\n", encoding="utf-8")
    result = subprocess.run(
        [sys.executable, str(HERE / "run_team.py"), "validate", str(source)],
        cwd=tmp_path, env={**os.environ, "PYTHONIOENCODING": "cp1252:strict"},
        capture_output=True, timeout=15,
    )
    assert result.returncode == 2
    diagnostic = result.stderr.decode("utf-8")
    assert "Configuration run stopped:" in diagnostic
    assert "Duplicate YAML key" in diagnostic and "🌱" in diagnostic
    assert "line 2" in diagnostic and "Traceback" not in diagnostic
    assert not (tmp_path / "agent_workspace_configured").exists()


def test_fixture_names_must_match_before_construction(config, responses, tmp_path):
    responses.pop("Writer")
    with pytest.raises(ValueError, match="exactly one response"):
        run_config(config, responses=responses, workspace=tmp_path / "not-created")
    assert not (tmp_path / "not-created").exists()


def test_digest_preserves_order_and_prompt_identity_but_ignores_yaml_formatting(config):
    data = config.model_dump()
    original = fingerprint(data)
    assert fingerprint(read_config(yaml.safe_dump(data, sort_keys=True)).model_dump()) == original
    reordered = deepcopy(data)
    reordered["agents"].reverse()
    assert fingerprint(reordered) != original
    changed = deepcopy(data)
    changed["agents"][1]["system_prompt"] += " Keep it brief."
    assert fingerprint(changed) != original


def test_replay_loads_stored_config_and_fixture_without_reading_original(tmp_path):
    source = tmp_path / "team.yaml"
    source.write_text((HERE / "team.yaml").read_text(encoding="utf-8"), encoding="utf-8")
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    assert main(["run", str(source), "--output", str(first), "--workspace", str(tmp_path / "runtime")]) == 0
    source.write_text("invalid changed source", encoding="utf-8")
    assert main(["replay", str(first), "--output", str(second), "--workspace", str(tmp_path / "runtime")]) == 0
    old, new = [json.loads(path.read_text(encoding="utf-8")) for path in (first, second)]
    assert old["config_sha256"] == new["config_sha256"]
    assert old["stages"] == new["stages"]
    modified = deepcopy(old)
    modified["config"]["task"] += " An unrecorded change."
    with pytest.raises(ValueError, match="digest"):
        replay_config(modified)
    modified = deepcopy(old)
    modified["native_agents"]["agents"][0]["max_loops"] = 9
    with pytest.raises(ValueError, match="execution policy"):
        replay_config(modified)


def test_empty_stage_is_a_failed_run_not_a_completed_note(config, responses, tmp_path):
    responses["Outline"] = ""
    report = run_config(config, responses=responses, workspace=tmp_path)
    assert report["status"] == "failed"
    assert report["stages"][0]["status"] == "no_response"
    assert report["final_output"] is None


def test_documented_run_and_replay_commands_with_fresh_processes(tmp_path):
    script = HERE / "run_team.py"
    first = tmp_path / "first.json"
    second = tmp_path / "replayed.json"
    commands = [
        ["run", str(HERE / "team.yaml"), "--output", str(first)],
        ["replay", str(first), "--output", str(second)],
    ]
    for arguments in commands:
        result = subprocess.run([sys.executable, str(script), *arguments], cwd=tmp_path,
                                env={**os.environ, "PYTHONIOENCODING": "cp1252"},
                                capture_output=True, timeout=45)
        assert result.returncode == 0, result.stderr.decode("cp1252", errors="replace")
        assert '"status": "complete"' in result.stdout.decode("utf-8")
    old, new = [json.loads(path.read_text(encoding="utf-8")) for path in (first, second)]
    assert old["config"] == new["config"]
    assert old["final_output"] == new["final_output"]
    assert "🌱" in new["final_output"]
