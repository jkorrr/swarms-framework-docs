"""Behavior checks using unmodified Swarms Agent and GraphWorkflow classes."""

from copy import deepcopy
import json
from pathlib import Path
import threading

import pytest

from pipeline import (
    MAX_BYTES, ScriptedModel, SourceNode, build_workflow, envelope,
    load_document, main, predecessor_data, run_pipeline, validate_output,
)

HERE = Path(__file__).parent


@pytest.fixture
def document():
    return load_document(HERE / "meeting.txt")


@pytest.fixture
def responses():
    return json.loads((HERE / "scripted-responses.json").read_text(encoding="utf-8"))


def make_models(responses):
    return {name: ScriptedModel(json.dumps(data)) for name, data in responses.items()}


def test_real_graph_and_agents_create_provenance_report(document, responses, tmp_path):
    models = make_models(responses)
    report = run_pipeline(document, models=models, workspace=tmp_path)
    assert report["mode"] == "scripted"
    assert report["complete"] is True
    assert report["source"] == document
    assert report["brief"] == responses["Brief"]
    assert report["stages"]["Actions"]["data"]["actions"][1]["owner"] is None
    assert report["stages"]["Actions"]["data"]["actions"][1]["due"] is None
    assert {name: len(model.calls) for name, model in models.items()} == {
        "Facts": 1, "Actions": 1, "Brief": 1}
    for name in ("Facts", "Actions"):
        transcript = str(models[name].calls[0])
        assert document["sha256"] in transcript
        assert "120 synthetic records" in transcript
    synthesis = str(models["Brief"].calls[0])
    assert "fact-3" in synthesis and "action-2" in synthesis


def test_fan_in_waits_for_both_branches_without_assuming_order(document, responses, tmp_path):
    actions_finished = threading.Event()
    order = []

    class OrderedModel(ScriptedModel):
        def __init__(self, name):
            super().__init__(json.dumps(responses[name]))
            self.name = name

        def run(self, **kwargs):
            if self.name == "Facts":
                assert actions_finished.wait(5), "Actions did not run concurrently"
            if self.name == "Brief":
                assert set(order) == {"Facts", "Actions"}
            result = super().run(**kwargs)
            order.append(self.name)
            if self.name == "Actions":
                actions_finished.set()
            return result

    models = {name: OrderedModel(name) for name in responses}
    report = run_pipeline(document, models=models, workspace=tmp_path)
    assert report["complete"] is True
    assert order == ["Actions", "Facts", "Brief"]


def test_invalid_branch_blocks_brief_but_preserves_other_branch(document, responses, tmp_path):
    responses["Facts"]["facts"][0]["evidence"][0]["line"] = 999
    models = make_models(responses)
    report = run_pipeline(document, models=models, workspace=tmp_path)
    assert report["complete"] is False and report["brief"] is None
    assert report["stages"]["Facts"]["status"] == "blocked"
    assert report["stages"]["Actions"]["status"] == "ok"
    assert report["stages"]["Brief"]["status"] == "blocked"
    assert models["Brief"].calls == []
    assert report["raw_responses"]["Facts"] == json.dumps(responses["Facts"])


def test_backend_failure_cannot_be_treated_as_a_valid_answer(document, responses, tmp_path):
    class BrokenModel:
        calls = 0

        def run(self, **kwargs):
            self.calls += 1
            raise RuntimeError("Scripted backend failure")

    models = make_models(responses)
    models["Facts"] = BrokenModel()
    report = run_pipeline(document, models=models, workspace=tmp_path)
    assert models["Facts"].calls == 1
    assert report["stages"]["Facts"]["status"] == "blocked"
    assert "did not record a model response" in report["stages"]["Facts"]["error"]
    assert models["Brief"].calls == []
    assert report["complete"] is False


def test_graph_error_string_is_rejected_by_downstream_stages(document, responses, tmp_path, monkeypatch):
    def broken_source(self, **kwargs):
        raise RuntimeError("Synthetic source failure")

    monkeypatch.setattr(SourceNode, "run", broken_source)
    models = make_models(responses)
    report = run_pipeline(document, models=models, workspace=tmp_path)
    assert report["complete"] is False
    assert {row["status"] for row in report["stages"].values()} == {"blocked"}
    assert all(model.calls == [] for model in models.values())


def test_separate_runs_do_not_reuse_agent_history(document, responses, tmp_path):
    graph1, stages1 = build_workflow(document, models=make_models(responses), workspace=tmp_path / "one")
    graph1.run("FIRST-DOCUMENT-ONLY")
    graph2, stages2 = build_workflow(document, models=make_models(responses), workspace=tmp_path / "two")
    graph2.run("SECOND-DOCUMENT-ONLY")
    from swarms import Agent, GraphWorkflow

    assert isinstance(graph1, GraphWorkflow) and isinstance(graph2, GraphWorkflow)
    for name in stages1:
        assert isinstance(stages1[name].agent, Agent)
        assert stages1[name].agent is not stages2[name].agent
        history = str(stages2[name].agent.short_memory.conversation_history)
        assert "SECOND-DOCUMENT-ONLY" in history
        assert "FIRST-DOCUMENT-ONLY" not in history


@pytest.mark.parametrize("change", ["bad_quote", "duplicate_id", "bool_line", "extra_field"])
def test_extraction_rejects_unsupported_evidence_and_contracts(document, responses, change):
    data = deepcopy(responses["Facts"])
    fact = data["facts"][0]
    if change == "bad_quote":
        fact["evidence"][0]["quote"] = "production data passed every test"
    elif change == "duplicate_id":
        data["facts"].append(deepcopy(fact))
    elif change == "bool_line":
        fact["evidence"][0]["line"] = True
    else:
        fact["confidence"] = 0.99
    with pytest.raises(ValueError):
        validate_output("Facts", json.dumps(data), {"Source": document})


@pytest.mark.parametrize("raw", [
    '{"facts":[],"facts":[]}', '```json\n{"facts":[]}\n```',
    '{"facts":NaN}', '{"facts":null}',
])
def test_invalid_json_is_not_repaired_into_success(document, raw):
    with pytest.raises(ValueError):
        validate_output("Facts", raw, {"Source": document})


@pytest.mark.parametrize("ids", [["fact-99"], ["fact-1", "fact-1"], []])
def test_synthesis_rejects_unknown_duplicate_or_no_item_references(responses, ids):
    brief = {"summary": "Test", "fact_ids": ids, "action_ids": []}
    with pytest.raises(ValueError):
        validate_output("Brief", json.dumps(brief), responses)


def test_predecessors_use_names_and_require_one_successful_envelope(document):
    valid = {"role": "user", "content": "Source: " + envelope("Source", data=document)}
    assert predecessor_data([valid], ("Source",)) == {"Source": document}
    for messages in ([], [valid, valid],
                     [{"role": "user", "content": "Source: " + envelope("Facts", data={})}],
                     [{"role": "user", "content": "Source: " + envelope("Source", error="Failed")}],
                     [{"role": "user", "content": "Source: [ERROR] Failed"}]):
        with pytest.raises(ValueError):
            predecessor_data(messages, ("Source",))


@pytest.mark.parametrize("raw", [b"", b" \n", b"\xff", b"text\x00", b"x" * (MAX_BYTES + 1)])
def test_bad_document_stops_before_graph_construction(tmp_path, monkeypatch, raw):
    def forbidden(*args, **kwargs):
        pytest.fail("Invalid document reached workflow construction")

    monkeypatch.setattr("pipeline.build_workflow", forbidden)
    document = tmp_path / "bad.txt"
    document.write_bytes(raw)
    output = tmp_path / "result.json"
    assert main(["--document", str(document), "--responses", str(HERE / "scripted-responses.json"),
                 "--output", str(output)]) == 2
    assert not output.exists()


def test_cli_writes_complete_and_blocked_reports(tmp_path, responses):
    output = tmp_path / "output.json"
    fixture = tmp_path / "responses.json"
    command = ["--document", str(HERE / "meeting.txt"), "--responses", str(fixture),
               "--output", str(output), "--workspace", str(tmp_path / "workspace")]
    fixture.write_text(json.dumps(responses), encoding="utf-8")
    assert main(command) == 0
    assert json.loads(output.read_text(encoding="utf-8"))["complete"] is True
    responses["Brief"]["fact_ids"] = ["fact-999"]
    fixture.write_text(json.dumps(responses), encoding="utf-8")
    assert main(command) == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["complete"] is False and report["brief"] is None
    assert report["stages"]["Facts"]["status"] == "ok"
