"""Exercise the local review workflow with real Swarms proposal generation."""

from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from catalog_review import (
    ScriptedModel, apply_reviewed, decide, main, preview, preview_text,
    proposal_digest, propose, read_catalog, read_proposal,
)

HERE = Path(__file__).parent


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


@pytest.fixture(scope="module")
def generated(tmp_path_factory):
    work = tmp_path_factory.mktemp("real-catalog-proposal")
    catalog = work / "catalog.json"
    catalog.write_bytes((HERE / "catalog.json").read_bytes())
    backend = ScriptedModel((HERE / "fixture-response.json").read_text(encoding="utf-8"))
    proposal = propose(catalog, work / "pending.json", llm=backend,
                       mode="scripted", workspace=work / "runtime")
    assert len(backend.calls) == 1
    return proposal, backend


@pytest.fixture
def files(tmp_path, generated):
    catalog = tmp_path / "catalog.json"
    catalog.write_bytes((HERE / "catalog.json").read_bytes())
    proposal = tmp_path / "pending.json"
    save(proposal, deepcopy(generated[0]))
    return catalog, proposal, tmp_path / "decision.json", tmp_path / "updated.json"


def approve(files, choice="approve"):
    catalog, proposal, decision, output = files
    _, _, reviewed_sha = preview(catalog, proposal)
    decide(proposal, decision, choice=choice, reviewer="test-editor",
           reviewed_sha256=reviewed_sha)


def test_real_agent_receives_catalog_and_creates_only_pending_changes(generated):
    proposal, backend = generated
    messages = str(backend.calls[0]["messages"])
    assert "MUG-01" in messages and "Canvas tote" in messages
    assert "price_cents" not in messages
    assert proposal["generation"]["mode"] == "scripted"
    assert proposal["generation"]["framework_version"] == "15.0.3"
    assert [change["sku"] for change in proposal["changes"]] == ["MUG-01", "BAG-03"]
    assert proposal["changes"][0]["before"] == "Blue mug. Ceramic. Holds 300 ml."
    assert "choice" not in proposal


def test_preview_and_approved_apply_preserve_input_and_other_fields(files):
    catalog, proposal, decision, output = files
    original_bytes = catalog.read_bytes()
    original, _ = read_catalog(catalog)
    view = preview_text(catalog, proposal)
    assert '"description": "Blue mug. Ceramic. Holds 300 ml."' in view
    assert '"description": "A blue ceramic mug with a 300 ml capacity."' in view
    assert proposal_digest(read_proposal(proposal)) in view
    assert not decision.exists() and not output.exists()
    approve(files)
    updated = apply_reviewed(*files)
    assert catalog.read_bytes() == original_bytes
    assert json.loads(output.read_text(encoding="utf-8")) == updated
    for before, after in zip(original["products"], updated["products"]):
        for key in ("sku", "name", "price_cents"):
            assert before[key] == after[key]
    assert updated["products"][1] == original["products"][1]


def test_pending_and_rejected_proposals_never_create_output(files):
    with pytest.raises(FileNotFoundError):
        apply_reviewed(*files)
    assert not files[3].exists()
    approve(files, choice="reject")
    with pytest.raises(ValueError, match="not approved"):
        apply_reviewed(*files)
    assert not files[3].exists()


def test_source_change_after_review_blocks_apply(files):
    approve(files)
    catalog, _, _, output = files
    value, _ = read_catalog(catalog)
    value["products"][0]["price_cents"] += 1
    save(catalog, value)
    with pytest.raises(ValueError, match="Source catalog changed"):
        apply_reviewed(*files)
    assert not output.exists()


@pytest.mark.parametrize("field", ["after", "reason"])
def test_proposal_edits_after_approval_require_a_new_decision(files, field):
    approve(files)
    proposal = read_proposal(files[1])
    proposal["changes"][0][field] += " changed"
    save(files[1], proposal)
    with pytest.raises(ValueError, match="does not cover"):
        apply_reviewed(*files)
    assert not files[3].exists()


def test_changed_preview_cannot_be_approved_with_an_old_digest(files):
    proposal = read_proposal(files[1])
    old_sha = proposal_digest(proposal)
    proposal["changes"][0]["after"] += " changed"
    save(files[1], proposal)
    with pytest.raises(ValueError, match="preview it again"):
        decide(files[1], files[2], choice="approve", reviewer="editor",
               reviewed_sha256=old_sha)
    assert not files[2].exists()


def test_review_decision_cannot_be_reused_for_a_different_proposal(files):
    approve(files)
    proposal = read_proposal(files[1])
    proposal["proposal_id"] = "different-proposal"
    save(files[1], proposal)
    with pytest.raises(ValueError, match="does not cover"):
        apply_reviewed(*files)


def test_existing_artifacts_and_source_are_never_overwritten(files):
    approve(files)
    decision_bytes = files[2].read_bytes()
    with pytest.raises(ValueError, match="already exists"):
        approve(files, choice="reject")
    assert files[2].read_bytes() == decision_bytes
    with pytest.raises(ValueError, match="separate path"):
        apply_reviewed(*files[:3], files[0])
    files[3].write_text("keep me", encoding="utf-8")
    with pytest.raises(ValueError, match="already exists"):
        apply_reviewed(*files)
    assert files[3].read_text(encoding="utf-8") == "keep me"


@pytest.mark.parametrize("bad_response", [
    "not JSON",
    '{"changes":[],"approved":true}',
    '{"changes":[]}',
    '{"changes":[{"sku":"missing","description":"new","reason":"why"}]}',
    '{"changes":[{"sku":"MUG-01","description":"new","reason":"why","price_cents":0}]}',
    '{"changes":[{"sku":"MUG-01","description":"new","reason":"why"},{"sku":"MUG-01","description":"other","reason":"why"}]}',
])
def test_bad_agent_proposals_fail_before_saving(tmp_path, bad_response):
    output = tmp_path / "pending.json"
    with pytest.raises(ValueError):
        propose(HERE / "catalog.json", output, llm=ScriptedModel(bad_response),
                mode="scripted", workspace=tmp_path / "runtime")
    assert not output.exists()


def test_real_agent_failure_does_not_leave_a_pending_proposal(tmp_path):
    class FailedModel:
        def run(self, **kwargs):
            raise RuntimeError("synthetic model failure")

    with pytest.raises(ValueError, match="no assistant response"):
        propose(HERE / "catalog.json", tmp_path / "pending.json", llm=FailedModel(),
                mode="scripted", workspace=tmp_path / "runtime")
    assert not (tmp_path / "pending.json").exists()


def test_catalog_schema_and_proposal_before_values_are_checked(files):
    catalog, proposal, _, _ = files
    body = read_proposal(proposal)
    body["changes"][0]["before"] = "not the original"
    save(proposal, body)
    with pytest.raises(ValueError, match="does not match"):
        preview(catalog, proposal)
    catalog.write_text('{"schema_version":1,"products":[],"products":[]}', encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate JSON"):
        read_catalog(catalog)


def test_cli_preview_decide_apply_and_error_exit_status(files, capsys):
    catalog, proposal, decision, output = map(str, files)
    assert main(["preview", "--catalog", catalog, "--proposal", proposal]) == 0
    assert "Reviewed SHA-256:" in capsys.readouterr().out
    assert main(["apply", "--catalog", catalog, "--proposal", proposal,
                 "--decision", decision, "--output", output]) == 2
    sha = proposal_digest(read_proposal(files[1]))
    assert main(["decide", "--proposal", proposal, "--decision", decision,
                 "--choice", "approve", "--reviewer", "test-editor",
                 "--reviewed-sha256", sha]) == 0
    assert main(["apply", "--catalog", catalog, "--proposal", proposal,
                 "--decision", decision, "--output", output]) == 0
    assert files[3].exists()
    assert "Review workflow error" in capsys.readouterr().err


def test_unicode_preview_uses_utf8_with_legacy_redirected_stdout(files):
    catalog, proposal, _, _ = files
    source, _ = read_catalog(catalog)
    source["products"][0]["description"] = "Café mug — 青い陶器"
    save(catalog, source)
    _, source_sha = read_catalog(catalog)
    pending = read_proposal(proposal)
    pending["source_sha256"] = source_sha
    pending["changes"][0]["before"] = source["products"][0]["description"]
    pending["changes"][0]["after"] = "青い陶器 — café mug"
    save(proposal, pending)
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "cp1252"
    completed = subprocess.run(
        [sys.executable, str(HERE / "catalog_review.py"), "preview",
         "--catalog", str(catalog), "--proposal", str(proposal)],
        env=environment, capture_output=True, check=False,
    )
    assert completed.returncode == 0, completed.stderr.decode("utf-8")
    assert "Café mug — 青い陶器" in completed.stdout.decode("utf-8")
    assert "青い陶器 — café mug" in completed.stdout.decode("utf-8")
