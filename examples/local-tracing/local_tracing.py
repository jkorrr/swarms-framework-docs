"""Application-owned, in-memory traces around real Swarms Agent calls."""

import argparse
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from typing import Any, Callable

from opentelemetry import trace
from opentelemetry.context import Context
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.sdk.trace.sampling import ALWAYS_ON
from opentelemetry.trace import Status, StatusCode

SPAN_OPTIONS = {"record_exception": False, "set_status_on_exception": False}
PROMPT = 'Summarize the supplied text. Return only JSON: {"summary": "..."}.'


def prepare_runtime(workspace: Path) -> None:
    """Call once before importing Swarms, in a fresh process."""
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
    if "swarms" in sys.modules:
        from swarms.telemetry.otel import swarm_telemetry
        if swarm_telemetry().ready:
            raise ValueError("Restart with native Swarms telemetry disabled")


def _require_prepared() -> None:
    if (os.getenv("SWARMS_TELEMETRY_ON") != "false"
            or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP") != "True"):
        raise ValueError("Call prepare_runtime before importing Swarms")
    if os.getenv("OTEL_SDK_DISABLED", "").lower().strip() == "true":
        raise ValueError("The OpenTelemetry SDK is disabled in this process")


class ScriptedModel:
    """Exercise Agent with fixed replies; no model quality is measured."""

    def __init__(self, response: str, *, fail: bool = False):
        self.response = response
        self.fail = fail

    def run(self, task=None, messages=None, **kwargs):
        if self.fail:
            raise RuntimeError(self.response)
        return self.response


def make_agent(llm: Any, *, model_name: str = "gpt-4o-mini") -> Any:
    _require_prepared()
    from swarms import Agent
    from swarms.telemetry.otel import swarm_telemetry
    if swarm_telemetry().ready:
        raise ValueError("Restart with native Swarms telemetry disabled")
    return Agent(
        agent_name="local-summary", system_prompt=PROMPT, llm=llm,
        model_name=model_name, max_loops=1, retry_attempts=1,
        output_type="final", print_on=False, autosave=False,
        persistent_memory=False, context_compression=False,
        dynamic_context_window=False, streaming_on=False, stream=False,
    )


def valid_summary(output: Any) -> bool:
    if not isinstance(output, str):
        return False

    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError("Duplicate key")
            result[key] = value
        return result

    try:
        data = json.loads(output, object_pairs_hook=object_pairs)
    except (ValueError, TypeError):
        return False
    return (isinstance(data, dict) and set(data) == {"summary"}
            and isinstance(data["summary"], str)
            and 0 < len(data["summary"].strip()) <= 240)


def _outcome(span, outcome: str) -> None:
    span.set_attribute("app.outcome", outcome)
    # No exception message, stack trace, prompt, or model output is recorded.
    span.set_status(Status(StatusCode.OK if outcome == "completed"
                           else StatusCode.ERROR))


def trace_jobs(tasks: list[str], *, agent_factory: Callable[[int], Any],
               workers: int = 2) -> dict:
    """Trace a bounded batch. Factory receives a zero-based task index.

    Prepare the runtime before this call; return a fresh Agent per task.
    Independent jobs deliberately start independent roots, even if the caller
    already has an active span. Only this provider's spans enter the report.
    """
    if (not isinstance(tasks, list) or not tasks or len(tasks) > 100
            or any(not isinstance(task, str) or not task.strip() for task in tasks)):
        raise ValueError("Provide 1 to 100 nonempty task strings")
    if type(workers) is not int or not 1 <= workers <= 16:
        raise ValueError("workers must be an integer between 1 and 16")
    _require_prepared()
    from swarms.telemetry.otel import swarm_telemetry
    if swarm_telemetry().ready:
        raise ValueError("Restart with native Swarms telemetry disabled")

    memory = InMemorySpanExporter()
    # Resource(...), rather than Resource.create(...), avoids auto-detected
    # process/host attributes and OTEL_RESOURCE_ATTRIBUTES for this provider.
    provider = TracerProvider(resource=Resource({"service.name": "local-agent-demo"}),
                              sampler=ALWAYS_ON, shutdown_on_exit=False)
    provider.add_span_processor(SimpleSpanProcessor(memory))
    tracer = provider.get_tracer("swarms-docs.local-tracing", "1.0")

    def run_one(index: int, task: str, root) -> dict:
        # use_span attaches the right root in this worker and restores the
        # worker's previous context on exit, including error paths.
        with trace.use_span(root, end_on_exit=True, **SPAN_OPTIONS):
            with tracer.start_as_current_span("agent.execute", **SPAN_OPTIONS) as execution:
                try:
                    agent = agent_factory(index)
                    before = len(agent.short_memory.conversation_history)
                    output = agent.run(task)
                    replied = any(row.get("role") == agent.agent_name for row in
                                  agent.short_memory.conversation_history[before:])
                    if not replied:
                        raise ValueError("No assistant reply")
                except Exception:
                    # Agent can also swallow a backend exception and return
                    # the input task; the history check above catches that.
                    _outcome(execution, "execution_error")
                    outcome = "execution_error"
                else:
                    _outcome(execution, "completed")
                    outcome = "completed"
            if outcome == "completed":
                with tracer.start_as_current_span("output.validate", **SPAN_OPTIONS) as validation:
                    outcome = "completed" if valid_summary(output) else "invalid_output"
                    _outcome(validation, outcome)
            _outcome(root, outcome)
            return {"job": index + 1, "outcome": outcome,
                    "trace_id": f"{root.get_span_context().trace_id:032x}"}

    roots = []
    try:
        with ThreadPoolExecutor(max_workers=workers) as pool:
            futures = []
            for index, task in enumerate(tasks):
                root = tracer.start_span("job", context=Context(),
                                         attributes={"app.job": index + 1})
                roots.append(root)
                futures.append(pool.submit(run_one, index, task, root))
            runs = [future.result() for future in futures]
        provider.force_flush()
        spans = []
        for span in sorted(memory.get_finished_spans(),
                           key=lambda item: (item.start_time, item.parent is not None)):
            spans.append({
                "name": span.name, "trace_id": f"{span.context.trace_id:032x}",
                "span_id": f"{span.context.span_id:016x}",
                "parent_id": f"{span.parent.span_id:016x}" if span.parent else None,
                "start_ns": span.start_time, "end_ns": span.end_time,
                "duration_ms": (span.end_time - span.start_time) / 1_000_000,
                "status": span.status.status_code.name,
                "attributes": dict(span.attributes),
            })
        return {"schema_version": 1, "mode": "application-spans",
                "framework_version": version("swarms"),
                "otel_sdk_version": version("opentelemetry-sdk"),
                "resource": dict(provider.resource.attributes),
                "runs": runs, "spans": spans}
    finally:
        # Covers submission/interrupt failures too. End already completed
        # spans only once; this is not cancellation of a running Agent.
        for root in roots:
            if root.is_recording():
                root.end()
        provider.shutdown()


def inspect_report(report: dict) -> str:
    if (not isinstance(report, dict) or report.get("schema_version") != 1
            or report.get("mode") != "application-spans"):
        raise ValueError("Unsupported trace report")
    lines = []
    for run in report["runs"]:
        lines.append(f"Job {run['job']} {run['outcome']} trace={run['trace_id']}")
        for span in report["spans"]:
            if span["trace_id"] == run["trace_id"]:
                indent = "  " if span["parent_id"] else ""
                lines.append(f"  {indent}{span['name']}: {span['status']} "
                             f"{span['duration_ms']:.3f} ms")
    return "\n".join(lines)


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run three synthetic Agent fixtures")
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workspace", type=Path, default=Path("trace-workspace"))
    run.add_argument("--workers", type=int, default=2)
    inspect = commands.add_parser("inspect", help="Read the local trace summary")
    inspect.add_argument("report", type=Path)
    args = parser.parse_args()
    try:
        if args.command == "inspect":
            report = json.loads(args.report.read_text(encoding="utf-8"))
        else:
            if args.output.exists():
                raise ValueError("Output already exists; choose a new report path")
            prepare_runtime(args.workspace)
            fixtures = [ScriptedModel('{"summary":"A blue mug holds 350 ml."}'),
                        ScriptedModel("not JSON"),
                        ScriptedModel("Synthetic backend failure", fail=True)]
            report = trace_jobs(["Summarize the fictional blue mug."] * 3,
                                agent_factory=lambda index: make_agent(fixtures[index]),
                                workers=args.workers)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            with args.output.open("x", encoding="utf-8", newline="\n") as handle:
                handle.write(json.dumps(report, indent=2) + "\n")
            print(f"Saved local traces to {args.output}")
        print(inspect_report(report))
        return 0
    except (OSError, ValueError, KeyError, TypeError):
        print("Unable to run or inspect traces; check arguments and the report path.",
              file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
