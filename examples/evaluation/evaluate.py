"""Collect and compare small, reproducible Swarms classification evaluations."""

import argparse
import hashlib
import json
import math
import os
import sys
from dataclasses import dataclass
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable

LABELS = ("billing", "technical", "account")
SCHEMA_VERSION = 1
PROMPT = (
    "Classify the support ticket as billing, technical, or account. "
    'Return only a JSON object with one key: {"label": "chosen label"}.'
)
AGENT_SETTINGS = {
    "max_loops": 1,
    "retry_attempts": 1,
    "output_type": "final",
    "autosave": False,
    "persistent_memory": False,
    "context_compression": False,
    "dynamic_context_window": False,
    "streaming_on": False,
    "stream": False,
    "print_on": False,
}
STATUSES = {"valid", "invalid_output", "execution_error"}


@dataclass(frozen=True)
class Case:
    id: str
    text: str
    expected: str


def digest(value: Any) -> str:
    """Hash canonical JSON so record order does not change a sorted manifest."""
    data = json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(data.encode("utf-8")).hexdigest()


def strict_json(text: str) -> Any:
    """Reject duplicate JSON keys and nonstandard NaN/Infinity constants."""
    def pairs(items: list[tuple[str, Any]]) -> dict:
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"Duplicate JSON key: {key}")
            result[key] = value
        return result

    def constant(value: str) -> None:
        raise ValueError(f"Nonstandard JSON constant: {value}")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=constant)


def validate_cases(cases: list[Case]) -> None:
    """Require a nonempty, uniquely identified dataset with known labels."""
    if not cases:
        raise ValueError("The dataset must contain at least one case")
    ids = set()
    for case in cases:
        if not isinstance(case, Case):
            raise ValueError("Each case must be a Case")
        if not isinstance(case.id, str) or not case.id.strip() or case.id in ids:
            raise ValueError("Case IDs must be nonempty and unique")
        if not isinstance(case.text, str) or not case.text.strip():
            raise ValueError(f"Case {case.id} needs nonempty text")
        if case.expected not in LABELS:
            raise ValueError(f"Case {case.id} has an unknown expected label")
        ids.add(case.id)


def load_cases(path: Path) -> list[Case]:
    """Read versioned JSONL without silently dropping blank or malformed rows."""
    cases = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        row = strict_json(line)
        if not isinstance(row, dict) or set(row) != {"schema_version", "id", "text", "expected"}:
            raise ValueError(f"Dataset line {line_number} has unexpected fields")
        if type(row["schema_version"]) is not int or row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Dataset line {line_number} has an unsupported version")
        cases.append(Case(row["id"], row["text"], row["expected"]))
    validate_cases(cases)
    return cases


def parse_label(output: Any) -> str:
    """Accept only the exact JSON label contract, without repairing responses."""
    if not isinstance(output, str):
        raise ValueError("Output must be JSON text")
    parsed = strict_json(output)
    if not isinstance(parsed, dict) or set(parsed) != {"label"}:
        raise ValueError("Output must have exactly the label key")
    if not isinstance(parsed["label"], str) or parsed["label"] not in LABELS:
        raise ValueError("Output label is not in the allowed set")
    return parsed["label"]


def prepare_framework(workspace: Path) -> None:
    """Configure workspace storage and LiteLLM's cost map before import."""
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"


def make_agent(name: str, *, llm: Any = None, model_name: str = "gpt-4o-mini", system_prompt: str = PROMPT) -> Any:
    """Create one fresh classifier; passing no llm selects the named provider."""
    from swarms import Agent

    return Agent(agent_name=name, llm=llm, model_name=model_name,
                 system_prompt=system_prompt, **AGENT_SETTINGS)


def metrics(results: list[dict]) -> dict:
    """Keep invalid and failed cases in the accuracy and recall denominators."""
    valid = sum(row["status"] == "valid" for row in results)
    correct = sum(row["status"] == "valid" and row["label"] == row["expected"] for row in results)
    recall = {}
    for label in LABELS:
        group = [row for row in results if row["expected"] == label]
        hits = sum(row["status"] == "valid" and row["label"] == label for row in group)
        recall[label] = hits / len(group) if group else None
    return {"total": len(results), "correct": correct, "valid": valid,
            "accuracy": correct / len(results), "valid_output_rate": valid / len(results),
            "recall_by_label": recall}


def evaluate_cases(cases: list[Case], agent_factory: Callable[[Case], Any], *,
                   variant: str, mode: str, workspace: Path,
                   batch_size: int = 4, max_workers: int = 2) -> dict:
    """Run real Agents and return one result for every supplied case.

    The factory receives each Case, but must not use its expected label to build
    the task or model response. Scripted fixtures deliberately test plumbing only.
    Use mode='provider' for an explicitly opted-in real-model experiment.
    """
    validate_cases(cases)
    if not isinstance(variant, str) or not variant.strip() or mode not in {"scripted", "provider"}:
        raise ValueError("Supply a variant and mode='scripted' or 'provider'")
    if any(type(n) is not int or n < 1 for n in (batch_size, max_workers)):
        raise ValueError("Batch size and max workers must be positive integers")
    prepare_framework(workspace)
    from swarms import Agent
    from swarms.structs.multi_agent_exec import run_agents_with_different_tasks

    agents = [agent_factory(case) for case in cases]
    if any(not isinstance(agent, Agent) for agent in agents) or len({id(a) for a in agents}) != len(agents):
        raise ValueError("The factory must create a fresh Swarms Agent for every case")
    for agent in agents:
        for key, value in AGENT_SETTINGS.items():
            if getattr(agent, key) != value:
                raise ValueError(f"Agent setting {key} must be {value!r} for this protocol")
    experiments = [{"model": a.model_name, "prompt_sha256": digest(a.system_prompt),
                    "temperature": a.temperature} for a in agents]
    if any(item != experiments[0] for item in experiments):
        raise ValueError("Use the same classifier configuration for every case in one run")
    starts = [len(a.short_memory.conversation_history) for a in agents]
    outputs = run_agents_with_different_tasks(
        [(agent, case.text) for agent, case in zip(agents, cases)],
        batch_size=batch_size, max_workers=max_workers,
    )
    if len(outputs) != len(cases):
        raise RuntimeError("The runner returned an unexpected result count")
    results = []
    for case, agent, start, output in zip(cases, agents, starts, outputs):
        row = {"id": case.id, "expected": case.expected, "status": "valid", "label": None,
               "raw_output": output if isinstance(output, str) else None}
        replies = [m for m in agent.short_memory.conversation_history[start:]
                   if m.get("role") == agent.agent_name]
        if isinstance(output, Exception) or not replies:
            row.update(status="execution_error", error=type(output).__name__ if isinstance(output, Exception)
                       else "NoAssistantResponse")
        else:
            try:
                row["label"] = parse_label(output)
            except (ValueError, TypeError):
                row.update(status="invalid_output", error="LabelContractViolation")
        results.append(row)
    manifest = sorted([{"id": c.id, "expected": c.expected, "text_sha256": digest(c.text)}
                       for c in cases], key=lambda item: item["id"])
    return {"schema_version": SCHEMA_VERSION, "variant": variant, "mode": mode,
            "notice": "Scripted outputs test the evaluation workflow, not model quality."
            if mode == "scripted" else "Provider results describe this dataset and configuration only.",
            "dataset": {"sha256": digest(manifest), "cases": manifest},
            "protocol": {"labels": list(LABELS), "parser": "exact-json-label-v1",
                         "framework": "swarms", "framework_version": version("swarms"),
                         "agent_settings": AGENT_SETTINGS.copy(), "batch_size": batch_size,
                         "max_workers": max_workers},
            "experiment": experiments[0], "results": results, "metrics": metrics(results)}


def validate_report(report: dict) -> list[dict]:
    """Validate saved results and recompute scores before applying any gates."""
    try:
        if type(report["schema_version"]) is not int or report["schema_version"] != SCHEMA_VERSION:
            raise ValueError("Unsupported report version")
        if report["mode"] not in {"scripted", "provider"}:
            raise ValueError("Unknown report mode")
        if not isinstance(report["variant"], str) or not report["variant"].strip():
            raise ValueError("Missing variant")
        experiment = report["experiment"]
        if not isinstance(experiment, dict) or set(experiment) != {"model", "prompt_sha256", "temperature"}:
            raise ValueError("Missing experiment metadata")
        if not isinstance(experiment["model"], str) or not experiment["model"].strip():
            raise ValueError("Missing model identifier")
        if not isinstance(experiment["prompt_sha256"], str) or len(experiment["prompt_sha256"]) != 64:
            raise ValueError("Invalid prompt digest")
        if type(experiment["temperature"]) not in (int, float) or not math.isfinite(experiment["temperature"]):
            raise ValueError("Invalid temperature metadata")
        protocol = report["protocol"]
        if set(protocol) != {"labels", "parser", "framework", "framework_version", "agent_settings", "batch_size", "max_workers"}:
            raise ValueError("Incomplete measurement protocol")
        if protocol["labels"] != list(LABELS) or protocol["parser"] != "exact-json-label-v1":
            raise ValueError("Unsupported labels or parser")
        if protocol["framework"] != "swarms" or not isinstance(protocol["framework_version"], str) or not protocol["framework_version"]:
            raise ValueError("Missing framework version")
        if protocol["agent_settings"] != AGENT_SETTINGS:
            raise ValueError("Unsupported Agent settings")
        if any(type(protocol[key]) is not int or protocol[key] < 1 for key in ("batch_size", "max_workers")):
            raise ValueError("Invalid runner settings")
        manifest = report["dataset"]["cases"]
        if not isinstance(manifest, list) or not manifest or manifest != sorted(manifest, key=lambda row: row["id"]):
            raise ValueError("Invalid dataset manifest")
        by_id = {}
        for case in manifest:
            if set(case) != {"id", "expected", "text_sha256"} or case["expected"] not in LABELS:
                raise ValueError("Invalid dataset case")
            if not isinstance(case["id"], str) or not case["id"].strip() or case["id"] in by_id:
                raise ValueError("Duplicate or missing case ID")
            if not isinstance(case["text_sha256"], str) or len(case["text_sha256"]) != 64:
                raise ValueError("Invalid case digest")
            by_id[case["id"]] = case
        if report["dataset"]["sha256"] != digest(manifest):
            raise ValueError("Dataset digest does not match its manifest")
        results = report["results"]
        if not isinstance(results, list) or len(results) != len(by_id):
            raise ValueError("Missing or extra evaluation results")
        seen = set()
        for row in results:
            if row["id"] not in by_id or row["id"] in seen or row["expected"] != by_id[row["id"]]["expected"]:
                raise ValueError("Results do not match the dataset manifest")
            if row["status"] not in STATUSES:
                raise ValueError("Unknown result status")
            if row["status"] == "valid" and row["label"] not in LABELS:
                raise ValueError("Valid result needs a known label")
            if row["status"] != "valid" and row["label"] is not None:
                raise ValueError("Failed results must not have a label")
            seen.add(row["id"])
        return results
    except (KeyError, TypeError, AttributeError) as error:
        raise ValueError("Malformed evaluation report") from error


def compare_reports(baseline: dict, candidate: dict, *, min_accuracy: float = 0.8,
                    min_valid_rate: float = 1.0, max_accuracy_drop: float = 0.0) -> dict:
    """Compare the same measurement conditions while allowing model/prompt changes."""
    limits = (min_accuracy, min_valid_rate, max_accuracy_drop)
    if any(isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x) or not 0 <= x <= 1 for x in limits):
        raise ValueError("Gate thresholds must be finite numbers between zero and one")
    old_rows, new_rows = validate_report(baseline), validate_report(candidate)
    for key in ("dataset", "protocol", "mode"):
        if baseline[key] != candidate[key]:
            raise ValueError(f"Reports have different {key}; rerun under matching conditions")
    old, new = metrics(old_rows), metrics(new_rows)
    gates = {"minimum_accuracy": new["accuracy"] >= min_accuracy,
             "minimum_valid_output_rate": new["valid_output_rate"] >= min_valid_rate,
             "maximum_accuracy_drop": new["accuracy"] >= old["accuracy"] - max_accuracy_drop}
    previous = {row["id"]: row for row in old_rows}
    changes = [{"id": row["id"], "expected": row["expected"],
                "baseline": {k: previous[row["id"]][k] for k in ("status", "label")},
                "candidate": {k: row[k] for k in ("status", "label")}}
               for row in new_rows if any(row[k] != previous[row["id"]][k] for k in ("status", "label"))]
    return {"passed": all(gates.values()), "mode": candidate["mode"],
            "notice": candidate.get("notice", ""), "gates": gates,
            "limits": {"min_accuracy": min_accuracy, "min_valid_rate": min_valid_rate,
                       "max_accuracy_drop": max_accuracy_drop},
            "baseline": {"variant": baseline["variant"], "experiment": baseline["experiment"], "metrics": old},
            "candidate": {"variant": candidate["variant"], "experiment": candidate["experiment"], "metrics": new},
            "changed_cases": changes}


class ScriptedModel:
    """A deterministic backend fixture, never an implementation of classification."""
    def __init__(self, response: str):
        self.response = response
        self.calls = 0

    def run(self, task: str | None = None, messages: list | None = None, **kwargs: Any) -> str:
        self.calls += 1
        return self.response


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="Run scripted fixtures through real Swarms Agents")
    run.add_argument("--cases", type=Path, required=True)
    run.add_argument("--responses", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--workspace", type=Path, default=Path(".swarms-evaluation-workspace"))
    compare = commands.add_parser("compare", help="Compare two saved evaluation reports")
    compare.add_argument("baseline", type=Path)
    compare.add_argument("candidate", type=Path)
    compare.add_argument("--min-accuracy", type=float, default=0.8)
    compare.add_argument("--min-valid-rate", type=float, default=1.0)
    compare.add_argument("--max-accuracy-drop", type=float, default=0.0)
    args = parser.parse_args(argv)
    try:
        if args.command == "run":
            cases = load_cases(args.cases)
            fixture = strict_json(args.responses.read_text(encoding="utf-8"))
            if not isinstance(fixture, dict) or set(fixture) != {"schema_version", "variant", "responses"}:
                raise ValueError("Invalid response fixture")
            if type(fixture["schema_version"]) is not int or fixture["schema_version"] != SCHEMA_VERSION:
                raise ValueError("Unsupported response fixture version")
            responses = fixture["responses"]
            if not isinstance(responses, dict) or set(responses) != {case.id for case in cases} or any(not isinstance(v, str) for v in responses.values()):
                raise ValueError("Response fixture must contain exactly one string for each case ID")
            report = evaluate_cases(cases, lambda case: make_agent(
                f"eval-{case.id}", llm=ScriptedModel(responses[case.id])),
                variant=fixture["variant"], mode="scripted", workspace=args.workspace)
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            print(json.dumps({"output": str(args.output), "notice": report["notice"], "metrics": report["metrics"]}, indent=2))
            return 0
        result = compare_reports(strict_json(args.baseline.read_text(encoding="utf-8")),
                                 strict_json(args.candidate.read_text(encoding="utf-8")),
                                 min_accuracy=args.min_accuracy, min_valid_rate=args.min_valid_rate,
                                 max_accuracy_drop=args.max_accuracy_drop)
        print(json.dumps(result, indent=2, ensure_ascii=False))
        return 0 if result["passed"] else 1
    except (ValueError, OSError) as error:
        print(f"Evaluation error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
