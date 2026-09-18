"""Evaluator checks, including real Swarms Agents and the real batch runner."""

from copy import deepcopy
import json
from pathlib import Path
import threading

import pytest

from evaluate import (
    Case, ScriptedModel, compare_reports, digest, evaluate_cases, load_cases,
    main, make_agent, parse_label, strict_json, validate_cases,
)

HERE = Path(__file__).parent


@pytest.fixture(scope="module")
def reports(tmp_path_factory):
    workspace = tmp_path_factory.mktemp("real-agent-evaluation")
    cases = load_cases(HERE / "cases.jsonl")
    result = []
    for filename in ("baseline-responses.json", "candidate-responses.json"):
        fixture = strict_json((HERE / filename).read_text(encoding="utf-8"))
        result.append(evaluate_cases(
            cases, lambda case: make_agent(case.id, llm=ScriptedModel(fixture["responses"][case.id])),
            variant=fixture["variant"], mode="scripted", workspace=workspace,
        ))
    return result


def test_real_agent_reports_and_default_gates(reports):
    baseline, candidate = reports
    assert baseline["metrics"]["accuracy"] == 1
    assert candidate["metrics"]["accuracy"] == pytest.approx(4 / 6)
    assert candidate["metrics"]["valid_output_rate"] == pytest.approx(5 / 6)
    assert candidate["metrics"]["recall_by_label"] == {"billing": 1, "technical": 0.5, "account": 0.5}
    assert candidate["results"][-1]["status"] == "invalid_output"
    assert candidate["results"][-1]["raw_output"] == "The label is account."
    assert compare_reports(baseline, baseline)["passed"] is True
    comparison = compare_reports(baseline, candidate)
    assert comparison["passed"] is False
    assert set(comparison["gates"].values()) == {False}
    assert {case["id"] for case in comparison["changed_cases"]} == {"technical-01", "account-02"}
    assert comparison["mode"] == "scripted"


@pytest.mark.parametrize("output", [
    '```json\n{"label":"billing"}\n```', '{"label":"Billing"}',
    '{"label":"unknown"}', '{"label":"billing","explanation":"x"}',
    '{"label":"account","label":"billing"}', '{"label":null}',
    '{"label":true}', '{"label":NaN}', '["billing"]', '{', None,
])
def test_output_contract_does_not_repair_responses(output):
    with pytest.raises((ValueError, TypeError)):
        parse_label(output)


def test_dataset_rejects_duplicate_ids_empty_cases_and_unknown_labels(tmp_path):
    case = Case("one", "ticket", "billing")
    for cases in ([], [case, case], [Case("", "ticket", "billing")],
                  [Case("one", " ", "billing")], [Case("one", "ticket", "other")]):
        with pytest.raises(ValueError):
            validate_cases(cases)
    path = tmp_path / "cases.jsonl"
    for text in ('{"schema_version":true,"id":"one","text":"x","expected":"billing"}',
                 '{"schema_version":1,"id":"one","text":"x","expected":"billing"}\n\n'):
        path.write_text(text, encoding="utf-8")
        with pytest.raises(ValueError):
            load_cases(path)


def test_real_runner_keeps_case_identity_when_completion_order_differs(tmp_path):
    second_finished = threading.Event()
    completion_order = []
    models = {}
    cases = [Case("first", "unique first ticket", "billing"),
             Case("second", "unique second ticket", "technical")]

    class OrderedModel:
        def __init__(self, case):
            self.case = case
            self.calls = []

        def run(self, task=None, messages=None, **kwargs):
            self.calls.append(deepcopy(messages))
            if self.case.id == "first":
                assert second_finished.wait(5), "Second model did not run concurrently"
            completion_order.append(self.case.id)
            if self.case.id == "second":
                second_finished.set()
            return json.dumps({"label": self.case.expected})

    def factory(case):
        model = models[case.id] = OrderedModel(case)
        return make_agent(case.id, llm=model)

    report = evaluate_cases(cases, factory, variant="ordered", mode="scripted", workspace=tmp_path)
    assert completion_order == ["second", "first"]
    assert [row["id"] for row in report["results"]] == ["first", "second"]
    assert report["metrics"]["correct"] == 2
    for case in cases:
        assert len(models[case.id].calls) == 1
        transcript = str(models[case.id].calls[0])
        assert case.text in transcript
        assert next(other.text for other in cases if other.id != case.id) not in transcript


def test_real_agent_swallowed_failure_cannot_score_the_input_as_an_answer(tmp_path):
    class ThrowingModel:
        calls = 0

        def run(self, **kwargs):
            self.calls += 1
            raise RuntimeError("synthetic backend failure")

    backend = ThrowingModel()
    cases = [Case("echo-trap", '{"label":"billing"}', "billing")]
    report = evaluate_cases(cases, lambda case: make_agent(case.id, llm=backend),
                            variant="failed", mode="scripted", workspace=tmp_path)
    assert backend.calls == 1
    assert report["results"][0]["status"] == "execution_error"
    assert report["results"][0]["label"] is None
    assert report["metrics"]["accuracy"] == 0
    assert report["metrics"]["valid_output_rate"] == 0
    assert report["metrics"]["recall_by_label"]["technical"] is None


def test_fresh_agent_is_required(tmp_path):
    shared = None

    def factory(case):
        nonlocal shared
        if shared is None:
            shared = make_agent("shared", llm=ScriptedModel('{"label":"billing"}'))
        return shared

    with pytest.raises(ValueError, match="fresh"):
        evaluate_cases([Case("one", "first", "billing"), Case("two", "second", "billing")],
                       factory, variant="shared", mode="scripted", workspace=tmp_path)


def test_dataset_fingerprint_is_independent_of_input_order(reports, tmp_path):
    cases = list(reversed(load_cases(HERE / "cases.jsonl")))
    fixture = strict_json((HERE / "baseline-responses.json").read_text(encoding="utf-8"))
    reordered = evaluate_cases(cases, lambda case: make_agent(case.id, llm=ScriptedModel(fixture["responses"][case.id])),
                               variant="reordered", mode="scripted", workspace=tmp_path)
    assert reordered["dataset"] == reports[0]["dataset"]
    assert compare_reports(reports[0], reordered)["passed"] is True


def test_model_and_prompt_may_change_but_results_join_by_id(reports):
    baseline = reports[0]
    candidate = deepcopy(baseline)
    candidate["variant"] = "different-model-prompt"
    candidate["experiment"] = {"model": "different-model", "prompt_sha256": digest("new prompt"), "temperature": 0.2}
    candidate["results"].reverse()
    assert compare_reports(baseline, candidate)["passed"] is True


@pytest.mark.parametrize("change", ["text", "truth", "protocol", "mode", "missing", "duplicate", "version"])
def test_incomparable_or_incomplete_reports_are_rejected(reports, change):
    baseline = reports[0]
    candidate = deepcopy(baseline)
    if change in {"text", "truth"}:
        first = candidate["dataset"]["cases"][0]
        if change == "text":
            first["text_sha256"] = digest("changed ticket")
        else:
            first["expected"] = "technical"
            next(row for row in candidate["results"] if row["id"] == first["id"])["expected"] = "technical"
        candidate["dataset"]["sha256"] = digest(candidate["dataset"]["cases"])
    elif change == "protocol":
        candidate["protocol"]["batch_size"] += 1
    elif change == "mode":
        candidate["mode"] = "provider"
    elif change == "missing":
        candidate["results"].pop()
    elif change == "duplicate":
        candidate["results"][-1] = candidate["results"][0]
    else:
        candidate["schema_version"] = True
    with pytest.raises(ValueError):
        compare_reports(baseline, candidate)


def test_scores_are_recomputed_and_thresholds_are_checked(reports):
    candidate = deepcopy(reports[1])
    candidate["metrics"] = {"accuracy": 1, "valid_output_rate": 1}
    assert compare_reports(reports[0], candidate)["passed"] is False
    for value in (float("nan"), float("inf"), -0.1, 1.1, True):
        with pytest.raises(ValueError):
            compare_reports(*reports, min_accuracy=value)
    assert compare_reports(*reports, min_accuracy=4 / 6, min_valid_rate=5 / 6,
                           max_accuracy_drop=1 / 3 + 1e-12)["passed"] is True


def test_cli_exit_codes_and_unchanged_reports(reports, tmp_path, capsys):
    paths = [tmp_path / "baseline.json", tmp_path / "candidate.json"]
    for report, path in zip(reports, paths):
        path.write_text(json.dumps(report), encoding="utf-8")
    before = [path.read_bytes() for path in paths]
    assert main(["compare", str(paths[0]), str(paths[0])]) == 0
    assert main(["compare", *(str(path) for path in paths)]) == 1
    assert main(["compare", *(str(path) for path in paths), "--min-accuracy", "nan"]) == 2
    assert [path.read_bytes() for path in paths] == before
    assert "Evaluation error" in capsys.readouterr().err
