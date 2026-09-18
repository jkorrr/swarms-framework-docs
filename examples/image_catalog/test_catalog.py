"""Exercise the real Agent multimodal path and batch helper without inference."""

import base64
from copy import deepcopy
import csv
from hashlib import sha256
import json
import os
from pathlib import Path
import subprocess
import sys
import threading
from types import SimpleNamespace

from PIL import Image
import pytest

import catalog


@pytest.fixture
def demo(tmp_path):
    root = tmp_path / "demo"
    catalog.make_demo(root)
    responses = json.loads((root / "responses.json").read_text(encoding="utf-8"))
    return root / "images", responses


def test_real_agent_images_remain_paired_after_reverse_completion(demo, tmp_path):
    folder, responses = demo
    captured, finished = [], []
    second_done = threading.Event()

    class ObservedModel:
        def __init__(self, name, response):
            self.name, self.response = name, response

        def run(self, task=None, img=None, messages=None, **kwargs):
            self.image = base64.b64decode(img.split(",", 1)[1])
            self.messages = deepcopy(messages)
            if self.name == "Caption-1":
                assert second_done.wait(5), "Second image never completed"
            finished.append(self.name)
            if self.name == "Caption-2":
                second_done.set()
            return self.response

    def factory(name, response, model):
        agent = catalog.make_agent(name, response, model)
        agent.llm = ObservedModel(name, response)
        captured.append(agent)
        return agent

    report = catalog.build_catalog(folder, responses=responses, workspace=tmp_path / "runtime", agent_factory=factory)
    from swarms import Agent

    assert finished == ["Caption-2", "Caption-1"]
    assert all(isinstance(agent, Agent) for agent in captured)
    assert captured[0].short_memory is not captured[1].short_memory
    assert [row["file"] for row in report["entries"]] == sorted(responses)
    assert all(row["status"] == "captioned" for row in report["entries"])
    for row, agent in zip(report["entries"], captured):
        expected = (folder / row["file"]).read_bytes()
        assert agent.llm.image == expected
        assert row["sha256"] == sha256(expected).hexdigest()
        assert row["caption"] == json.loads(responses[row["file"]])["caption"]
        assert row["raw_output"] == responses[row["file"]]
        assert sum(message.get("content") == catalog.PROMPT for message in agent.llm.messages) == 1
        blocks = [block for message in agent.llm.messages if isinstance(message.get("content"), list)
                  for block in message["content"] if block.get("type") == "image_url"]
        assert len(blocks) == 1
        assert base64.b64decode(blocks[0]["image_url"]["url"].split(",", 1)[1]) == expected


def test_real_litellm_request_contains_the_image_without_provider_call(demo, tmp_path, monkeypatch):
    folder, _ = demo
    os.environ["WORKSPACE_DIR"] = str(tmp_path / "runtime")
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    import swarms.utils.litellm_wrapper as wrapper

    requests = []

    def completion(**kwargs):
        requests.append(deepcopy(kwargs))
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content='{"caption": "Fixture reply."}'))])

    monkeypatch.setattr(wrapper, "completion", completion)
    _, image = catalog.inspect_image(folder / "red.png")
    agent = catalog.make_agent("Provider-preparation", None, "openai/gpt-4o")
    assert isinstance(agent.llm, wrapper.LiteLLM)
    assert catalog.ImageTask(agent, image).run(catalog.PROMPT) == '{"caption": "Fixture reply."}'
    assert len(requests) == 1
    blocks = [block for message in requests[0]["messages"] if isinstance(message.get("content"), list)
              for block in message["content"] if block.get("type") == "image_url"]
    assert [block["image_url"]["url"] for block in blocks] == [image]
    assert any(message.get("content") == catalog.PROMPT for message in requests[0]["messages"])


def test_invalid_image_is_listed_and_never_reaches_an_agent(demo, tmp_path):
    folder, responses = demo
    (folder / "broken.png").write_bytes((folder / "red.png").read_bytes()[:-12])
    (folder / "notes.txt").write_text("not an image", encoding="utf-8")
    called = []

    def factory(*args):
        called.append(args[0])
        return catalog.make_agent(*args)

    report = catalog.build_catalog(folder, responses=responses, workspace=tmp_path / "runtime", agent_factory=factory)
    entries = {row["file"]: row for row in report["entries"]}
    assert entries["broken.png"]["status"] == "invalid_image"
    assert entries["notes.txt"]["status"] == "skipped"
    assert len(called) == 2 and report["status"] == "partial"
    with pytest.raises(ValueError, match="partial"):
        catalog.export_csv(report, tmp_path / "refused.csv")
    assert not (tmp_path / "refused.csv").exists()
    assert catalog.export_csv(report, tmp_path / "partial.csv", allow_partial=True) == 2


@pytest.mark.parametrize("response", ["", "not JSON", '{"caption": " "}', '{"caption": "ok", "extra": true}'])
def test_invalid_caption_is_not_exportable(demo, tmp_path, response):
    folder, responses = demo
    responses["red.png"] = response
    report = catalog.build_catalog(folder, responses=responses, workspace=tmp_path / "runtime")
    row = next(row for row in report["entries"] if row["file"] == "red.png")
    assert row["status"] == "invalid_caption" and "caption" not in row
    assert report["status"] == "partial"


def test_real_agent_failure_does_not_turn_task_echo_into_a_caption(demo, tmp_path, monkeypatch):
    folder, responses = demo
    monkeypatch.setattr(catalog, "PROMPT", '{"caption": "This is only the input task"}')

    class ThrowingModel:
        def run(self, **kwargs):
            raise RuntimeError("deliberate fixture failure")

    def factory(*args):
        agent = catalog.make_agent(*args)
        agent.llm = ThrowingModel()
        return agent

    report = catalog.build_catalog(folder, responses=responses, workspace=tmp_path / "runtime", agent_factory=factory)
    assert all(row["status"] == "model_error" for row in report["entries"])
    with pytest.raises(ValueError, match="no usable captions"):
        catalog.export_csv(report, tmp_path / "empty.csv", allow_partial=True)


def test_image_input_is_a_snapshot_after_source_changes(demo, tmp_path):
    folder, responses = demo
    original = (folder / "red.png").read_bytes()
    meta, image = catalog.inspect_image(folder / "red.png")
    (folder / "red.png").write_bytes(b"changed source")
    assert base64.b64decode(image.split(",", 1)[1]) == original
    assert meta["sha256"] == sha256(original).hexdigest()


@pytest.mark.parametrize("kind", ["bytes", "pixels", "animation"])
def test_image_limits_reject_before_inference(demo, monkeypatch, kind):
    folder, _ = demo
    path = folder / "red.png"
    if kind == "bytes":
        monkeypatch.setattr(catalog, "MAX_BYTES", 10)
    elif kind == "pixels":
        monkeypatch.setattr(catalog, "MAX_PIXELS", 10)
    else:
        Image.new("RGB", (32, 32), "red").save(path, save_all=True, append_images=[Image.new("RGB", (32, 32), "blue")])
    with pytest.raises(ValueError):
        catalog.inspect_image(path)


@pytest.mark.parametrize("change", ["duplicate", "status", "caption", "digest", "dimensions", "summary"])
def test_export_rejects_inconsistent_manifest(demo, tmp_path, change):
    folder, responses = demo
    # Use a synthetic artifact here: SDK execution is covered separately.
    report = {"manifest_version": 1, "mode": "scripted", "status": "complete", "entries": [
        {"file": "red.png", "status": "captioned", "sha256": "a" * 64, "width": 32, "height": 32, "caption": "Red."}
    ]}
    row = report["entries"][0]
    if change == "duplicate":
        report["entries"].append(deepcopy(row))
    elif change == "status":
        row["status"] = "unknown"
    elif change == "caption":
        row["caption"] = " "
    elif change == "digest":
        row["sha256"] = "missing"
    elif change == "dimensions":
        row["width"] = True
    else:
        report["status"] = "partial"
    with pytest.raises(ValueError):
        catalog.export_csv(report, tmp_path / "bad.csv")
    assert not (tmp_path / "bad.csv").exists()


def test_documented_cli_demo_build_export_unicode_and_partial(tmp_path):
    script = Path(catalog.__file__)
    env = {**os.environ, "PYTHONIOENCODING": "cp1252:strict"}

    def command(*arguments, expected=0):
        result = subprocess.run([sys.executable, str(script), *arguments], cwd=tmp_path,
                                env=env, capture_output=True, timeout=45)
        assert result.returncode == expected, result.stderr.decode("utf-8", errors="replace")
        return result

    command("demo", "sample")
    command("build", "sample/images", "--responses", "sample/responses.json", "--output", "catalog.json")
    command("export", "catalog.json", "--output", "captions.csv")
    with (tmp_path / "captions.csv").open(encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 2 and "café" in rows[0]["caption"] and "🌱" in rows[0]["file"]
    (tmp_path / "sample/images/broken-🌱.png").write_bytes(b"broken")
    command("build", "sample/images", "--responses", "sample/responses.json", "--output", "partial.json", expected=1)
    refused = command("export", "partial.json", "--output", "refused.csv", expected=2)
    assert "partial" in refused.stderr.decode("utf-8")
    assert not (tmp_path / "refused.csv").exists()
    command("export", "partial.json", "--output", "partial.csv", "--allow-partial")
    bad = command("build", "missing-🌱", "--responses", "sample/responses.json", "--output", "bad.json", expected=2)
    assert "missing-🌱" in bad.stderr.decode("utf-8")


@pytest.mark.parametrize("payload", [None, []])
def test_cli_rejects_nonobject_responses_before_importing_swarms(demo, tmp_path, payload):
    folder, _ = demo
    source = tmp_path / "invalid-responses.json"
    source.write_text(json.dumps(payload), encoding="utf-8")
    code = (
        "import sys; import catalog; result = catalog.main(sys.argv[1:]); "
        "assert 'swarms' not in sys.modules, 'Invalid fixture imported Swarms'; "
        "raise SystemExit(result)"
    )
    result = subprocess.run(
        [sys.executable, "-c", code, "build", str(folder), "--responses", str(source),
         "--output", str(tmp_path / "no-report.json")],
        cwd=tmp_path, capture_output=True, timeout=15,
        env={**os.environ, "PYTHONPATH": str(Path(catalog.__file__).parent) + os.pathsep + os.environ.get("PYTHONPATH", ""),
             "PYTHONIOENCODING": "cp1252:strict"},
    )
    assert result.returncode == 2, result.stderr.decode("utf-8", errors="replace")
    assert "object mapping" in result.stderr.decode("utf-8")
    assert not (tmp_path / "no-report.json").exists()


@pytest.mark.parametrize("field", ["mode", "entry_status"])
def test_cli_invalid_manifest_types_are_validation_errors(tmp_path, field):
    report = {"manifest_version": 1, "mode": "scripted", "status": "complete",
              "entries": [{"file": "red.png", "status": "skipped"}]}
    if field == "mode":
        report["mode"] = []
    else:
        report["entries"][0]["status"] = []
    source = tmp_path / "invalid.json"
    source.write_text(json.dumps(report), encoding="utf-8")
    output = tmp_path / "no-output.csv"
    result = subprocess.run(
        [sys.executable, catalog.__file__, "export", str(source), "--output", str(output)],
        cwd=tmp_path, env={**os.environ, "PYTHONIOENCODING": "cp1252:strict"},
        capture_output=True, timeout=15,
    )
    assert result.returncode == 2
    assert "Catalog stopped:" in result.stderr.decode("utf-8")
    assert "Traceback" not in result.stderr.decode("utf-8")
    assert not output.exists()
