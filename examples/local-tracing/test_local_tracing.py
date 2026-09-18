"""Real Agent and real OTel exporter tests, using synthetic model backends."""

import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import threading

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
import pytest

import local_tracing as demo

HERE = Path(__file__).parent


@pytest.fixture(autouse=True)
def runtime(tmp_path, monkeypatch):
    demo.prepare_runtime(tmp_path / "runtime")
    # Fail any accidental socket connection during this test process, including
    # import-time initialization. This is a test assertion, not app isolation.
    attempts = []

    def forbidden(*args, **kwargs):
        attempts.append(True)
        raise AssertionError("Unexpected network connection")

    monkeypatch.setattr(socket.socket, "connect", forbidden)
    yield
    assert not attempts


def factory(responses, *, failures=()):
    return lambda index: demo.make_agent(demo.ScriptedModel(
        responses[index], fail=index in failures))


def test_real_agent_three_outcomes_and_complete_local_span_tree():
    report = demo.trace_jobs(["Synthetic mug"] * 3,
                            agent_factory=factory(['{"summary":"Blue mug"}',
                                                   "not JSON", "backend broke"],
                                                  failures=(2,)), workers=3)
    assert report["framework_version"] == "15.0.3"
    assert report["otel_sdk_version"] == "1.44.0"
    assert [run["outcome"] for run in report["runs"]] == [
        "completed", "invalid_output", "execution_error"]
    assert len(report["spans"]) == 8
    assert len({run["trace_id"] for run in report["runs"]}) == 3
    for run in report["runs"]:
        spans = [s for s in report["spans"] if s["trace_id"] == run["trace_id"]]
        root, = [s for s in spans if s["parent_id"] is None]
        assert root["name"] == "job"
        assert root["attributes"] == {"app.job": run["job"], "app.outcome": run["outcome"]}
        assert root["status"] == ("OK" if run["outcome"] == "completed" else "ERROR")
        for child in [s for s in spans if s["parent_id"]]:
            assert child["parent_id"] == root["span_id"]
            assert root["start_ns"] <= child["start_ns"] <= child["end_ns"] <= root["end_ns"]
        assert all(s["duration_ms"] >= 0 for s in spans)


def test_swallowed_backend_failure_does_not_accept_echoed_valid_task():
    report = demo.trace_jobs(['{"summary":"This input is not an assistant reply"}'],
                            agent_factory=factory(["provider error"], failures=(0,)))
    assert report["runs"][0]["outcome"] == "execution_error"
    assert [s["name"] for s in report["spans"]] == ["job", "agent.execute"]


def test_in_memory_spans_omit_payloads_exception_events_and_environment(monkeypatch):
    marker = "SYNTHETIC-PAYLOAD-DO-NOT-EXPORT"
    monkeypatch.setenv("OTEL_RESOURCE_ATTRIBUTES", f"task={marker}")
    exported = []
    original = demo.InMemorySpanExporter

    class ObservedExporter(original):
        def export(self, spans):
            exported.extend(spans)
            return super().export(spans)

    monkeypatch.setattr(demo, "InMemorySpanExporter", ObservedExporter)
    report = demo.trace_jobs([marker] * 2,
                            agent_factory=factory([json.dumps({"summary": marker}),
                                                   marker], failures=(1,)))
    assert marker not in json.dumps(report)
    assert exported
    for span in exported:
        assert not span.events and not span.links
        assert span.status.description is None
        assert set(span.attributes) <= {"app.job", "app.outcome"}
        assert dict(span.resource.attributes) == {"service.name": "local-agent-demo"}
        assert marker not in span.to_json()


def test_concurrent_workers_have_distinct_context_and_caller_is_restored():
    barrier = threading.Barrier(2)
    seen = []
    caller_provider = TracerProvider(shutdown_on_exit=False)
    global_provider = trace.get_tracer_provider()

    class ConcurrentModel(demo.ScriptedModel):
        def run(self, **kwargs):
            seen.append(trace.get_current_span().get_span_context())
            barrier.wait(timeout=5)
            return super().run(**kwargs)

    try:
        with caller_provider.get_tracer("test").start_as_current_span("unrelated") as caller:
            report = demo.trace_jobs(["first", "second"], workers=2,
                                    agent_factory=lambda _: demo.make_agent(
                                        ConcurrentModel('{"summary":"Synthetic"}')))
            assert all(run["outcome"] == "completed" for run in report["runs"])
            assert trace.get_current_span() is caller
            assert all(r["trace_id"] != f"{caller.get_span_context().trace_id:032x}"
                       for r in report["runs"])
        assert not trace.get_current_span().get_span_context().is_valid
    finally:
        caller_provider.shutdown()
    assert trace.get_tracer_provider() is global_provider
    assert len({s.trace_id for s in seen}) == 2
    assert all(s.is_valid for s in seen)
    assert {f"{s.trace_id:032x}" for s in seen} == {r["trace_id"] for r in report["runs"]}


def test_reused_worker_context_is_restored_after_failure_and_next_batch():
    # Three jobs on one worker exercise reuse after both success and failure.
    before = trace.get_current_span()
    first = demo.trace_jobs(["a", "b", "c"], workers=1,
                            agent_factory=factory(['{"summary":"a"}', "error",
                                                   '{"summary":"c"}'], failures=(1,)))
    second = demo.trace_jobs(["d"], agent_factory=factory(['{"summary":"d"}']))
    assert trace.get_current_span() is before
    roots = [s for s in first["spans"] + second["spans"] if s["name"] == "job"]
    assert len(roots) == 4 and all(s["parent_id"] is None for s in roots)
    assert len({s["trace_id"] for s in roots}) == 4
    assert len(second["spans"]) == 3


@pytest.mark.parametrize("output", [None, {}, "", "{}", '{"summary":""}',
                                    '{"summary":42}', '{"summary":"ok","extra":1}',
                                    '{"summary":"a","summary":"b"}',
                                    json.dumps({"summary": "x" * 241})])
def test_invalid_output_contract(output):
    assert not demo.valid_summary(output)


@pytest.mark.parametrize("tasks,workers", [([], 1), ([""], 1), ([42], 1),
                                          (["a"] * 101, 1), (["a"], 0),
                                          (["a"], 17), (["a"], True)])
def test_invalid_batch_does_not_construct_agents(tasks, workers):
    def unexpected(_):
        pytest.fail("Agent should not be constructed")
    with pytest.raises(ValueError):
        demo.trace_jobs(tasks, agent_factory=unexpected, workers=workers)


def test_factory_exception_is_recorded_without_details():
    def broken_factory(_):
        raise RuntimeError("SYNTHETIC-PRIVATE-ERROR")
    report = demo.trace_jobs(["synthetic"], agent_factory=broken_factory)
    assert report["runs"][0]["outcome"] == "execution_error"
    assert "SYNTHETIC-PRIVATE-ERROR" not in json.dumps(report)


def test_native_exporter_is_not_created(monkeypatch):
    from swarms.telemetry import otel
    assert not otel.swarm_telemetry().ready
    def unexpected(*args, **kwargs):
        pytest.fail("Native HTTP exporter must not be constructed")
    monkeypatch.setattr(otel, "OTLPSpanExporter", unexpected)
    assert not otel.SwarmTelemetry().ready
    report = demo.trace_jobs(["Synthetic"], agent_factory=factory(['{"summary":"ok"}']))
    assert report["runs"][0]["outcome"] == "completed"


def test_unprepared_or_disabled_runtime_is_rejected(monkeypatch):
    monkeypatch.setenv("OTEL_SDK_DISABLED", " true ")
    with pytest.raises(ValueError):
        demo.trace_jobs(["a"], agent_factory=factory(["a"]))
    monkeypatch.delenv("OTEL_SDK_DISABLED")
    monkeypatch.setenv("SWARMS_TELEMETRY_ON", "true")
    with pytest.raises(ValueError):
        demo.make_agent(demo.ScriptedModel("unused"))


def test_cli_run_inspect_utf8_and_existing_output(tmp_path):
    report = tmp_path / "tracés.json"
    env = dict(os.environ, PYTHONIOENCODING="cp1252")
    args = [sys.executable, str(HERE / "local_tracing.py")]
    run_args = ["run", "--output", str(report), "--workspace", str(tmp_path / "cli-runtime")]
    run = subprocess.run(args + run_args, env=env, capture_output=True, timeout=60)
    assert run.returncode == 0, run.stderr.decode("utf-8", errors="replace")
    text = run.stdout.decode("utf-8")
    assert "tracés.json" in text
    assert "Job 1 completed" in text and "Job 2 invalid_output" in text
    assert "Job 3 execution_error" in text
    saved = report.read_bytes()
    inspect = subprocess.run(args + ["inspect", str(report)], env=env,
                             capture_output=True, timeout=20)
    assert inspect.returncode == 0
    assert inspect.stdout.decode("utf-8").splitlines() == demo.inspect_report(json.loads(saved)).splitlines()
    again = subprocess.run(args + run_args, env=env, capture_output=True, timeout=20)
    assert again.returncode == 2 and report.read_bytes() == saved


@pytest.mark.parametrize("value", [[], None, "wrong shape"])
def test_cli_inspect_rejects_nonobject_json(tmp_path, monkeypatch, capsys, value):
    path = tmp_path / "malformed.json"
    path.write_text(json.dumps(value), encoding="utf-8")
    monkeypatch.setattr(sys, "argv", ["local_tracing.py", "inspect", str(path)])
    assert demo.main() == 2
    assert "Unable to run or inspect" in capsys.readouterr().err
