"""Propose, review, and apply description edits to a synthetic local catalog."""

import argparse
from copy import deepcopy
from datetime import datetime, timezone
import difflib
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import sys
from typing import Any
import uuid

PROMPT = (
    "Suggest clearer descriptions for the supplied catalog. Preserve its facts; "
    "do not invent features. Return JSON with exactly one changes array. Each "
    "entry must contain sku, description, and reason, all strings. Suggest only "
    "description edits. Do not approve, apply, or write anything."
)


def _object(pairs: list[tuple[str, Any]]) -> dict:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def parse_json(text: str) -> Any:
    def invalid_constant(value: str) -> None:
        raise ValueError(f"Invalid JSON constant: {value}")

    return json.loads(text, object_pairs_hook=_object,
                      parse_constant=invalid_constant)


def _keys(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} must have exactly these fields: {sorted(keys)}")


def _text(value: Any, name: str, limit: int = 500) -> None:
    if not isinstance(value, str) or not value.strip() or len(value) > limit:
        raise ValueError(f"{name} must be nonempty text of at most {limit} characters")


def _sha(value: Any) -> None:
    if (not isinstance(value, str) or len(value) != 64
            or any(c not in "0123456789abcdef" for c in value)):
        raise ValueError("Invalid SHA-256 value")


def _schema(value: Any) -> None:
    if type(value) is not int or value != 1:
        raise ValueError("Only schema_version=1 is supported")


def proposal_digest(proposal: dict) -> str:
    canonical = json.dumps(proposal, sort_keys=True, ensure_ascii=False,
                           separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def read_catalog(path: Path) -> tuple[dict, str]:
    raw = path.read_bytes()
    catalog = parse_json(raw.decode("utf-8"))
    _keys(catalog, {"schema_version", "products"}, "Catalog")
    _schema(catalog["schema_version"])
    if not isinstance(catalog["products"], list) or not catalog["products"]:
        raise ValueError("Catalog products must be a nonempty list")
    seen = set()
    for product in catalog["products"]:
        _keys(product, {"sku", "name", "description", "price_cents"}, "Product")
        for key in ("sku", "name", "description"):
            _text(product[key], key)
        if product["sku"] in seen:
            raise ValueError("Catalog SKUs must be unique")
        if type(product["price_cents"]) is not int or product["price_cents"] < 0:
            raise ValueError("price_cents must be a nonnegative integer")
        seen.add(product["sku"])
    return catalog, hashlib.sha256(raw).hexdigest()


def read_proposal(path: Path) -> dict:
    proposal = parse_json(path.read_text(encoding="utf-8"))
    _keys(proposal, {"schema_version", "proposal_id", "source_sha256",
                     "generation", "changes"}, "Proposal")
    _schema(proposal["schema_version"])
    _text(proposal["proposal_id"], "proposal_id", 100)
    _sha(proposal["source_sha256"])
    generation = proposal["generation"]
    _keys(generation, {"mode", "framework_version", "model_name"}, "Generation")
    if generation["mode"] not in ("scripted", "provider"):
        raise ValueError("Unknown generation mode")
    _text(generation["framework_version"], "framework_version", 100)
    _text(generation["model_name"], "model_name", 100)
    changes = proposal["changes"]
    if not isinstance(changes, list) or not changes:
        raise ValueError("A proposal must contain at least one change")
    seen = set()
    for change in changes:
        _keys(change, {"sku", "before", "after", "reason"}, "Change")
        for key in change:
            _text(change[key], key)
        if change["sku"] in seen or change["before"] == change["after"]:
            raise ValueError("Changes must have unique SKUs and modify descriptions")
        seen.add(change["sku"])
    return proposal


def write_new_json(path: Path, data: dict) -> None:
    """Create an artifact without replacing an existing file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        handle.write(json.dumps(data, indent=2, ensure_ascii=False) + "\n")


def _separate(output: Path, *inputs: Path) -> None:
    if any(output.resolve() == path.resolve() for path in inputs):
        raise ValueError("Output must be a separate path from every input")
    if output.exists():
        raise ValueError("Output already exists; choose a new artifact path")


class ScriptedModel:
    """Return a fixture through the real Agent; this is not a language model."""
    def __init__(self, response: str):
        self.response = response
        self.calls = []

    def run(self, task=None, messages=None, **kwargs):
        self.calls.append({"task": task, "messages": deepcopy(messages)})
        return self.response


def propose(catalog_path: Path, proposal_path: Path, *, llm: Any,
            mode: str, workspace: Path,
            model_name: str = "gpt-4o-mini") -> dict:
    """Ask a real Agent for edits; save a pending artifact only after validation."""
    _separate(proposal_path, catalog_path)
    catalog, source_sha = read_catalog(catalog_path)
    if mode not in ("scripted", "provider"):
        raise ValueError("Mode must be scripted or provider")
    if mode == "scripted" and llm is None:
        raise ValueError("Scripted mode requires a custom backend")
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    from swarms import Agent

    agent = Agent(
        agent_name="catalog-proposer", system_prompt=PROMPT, llm=llm,
        model_name=model_name, max_loops=1, retry_attempts=1,
        output_type="final", print_on=False, autosave=False,
        persistent_memory=False, context_compression=False,
        dynamic_context_window=False, streaming_on=False, stream=False,
    )
    products = [{key: item[key] for key in ("sku", "name", "description")}
                for item in catalog["products"]]
    start = len(agent.short_memory.conversation_history)
    answer = agent.run(json.dumps({"products": products}, ensure_ascii=False))
    if not any(row.get("role") == agent.agent_name
               for row in agent.short_memory.conversation_history[start:]):
        raise ValueError("Agent produced no assistant response")
    if not isinstance(answer, str):
        raise ValueError("Agent response must be JSON text")
    response = parse_json(answer)
    _keys(response, {"changes"}, "Agent response")
    if not isinstance(response["changes"], list) or not response["changes"]:
        raise ValueError("Agent must propose at least one description change")
    by_sku = {product["sku"]: product for product in catalog["products"]}
    changes, seen = [], set()
    for edit in response["changes"]:
        _keys(edit, {"sku", "description", "reason"}, "Proposed edit")
        for key in edit:
            _text(edit[key], key)
        sku = edit["sku"]
        if sku not in by_sku or sku in seen:
            raise ValueError("Agent proposed an unknown or repeated SKU")
        if edit["description"] == by_sku[sku]["description"]:
            raise ValueError("Agent proposed an unchanged description")
        changes.append({"sku": sku, "before": by_sku[sku]["description"],
                        "after": edit["description"], "reason": edit["reason"]})
        seen.add(sku)
    proposal = {
        "schema_version": 1, "proposal_id": uuid.uuid4().hex,
        "source_sha256": source_sha,
        "generation": {"mode": mode, "framework_version": version("swarms"),
                       "model_name": model_name},
        "changes": changes,
    }
    write_new_json(proposal_path, proposal)
    return proposal


def preview(catalog_path: Path, proposal_path: Path) -> tuple[dict, dict, str]:
    """Check the source revision and produce the exact description-only result."""
    catalog, source_sha = read_catalog(catalog_path)
    proposal = read_proposal(proposal_path)
    if proposal["source_sha256"] != source_sha:
        raise ValueError("Source catalog changed; generate and review a new proposal")
    by_sku = {product["sku"]: product for product in catalog["products"]}
    for change in proposal["changes"]:
        product = by_sku.get(change["sku"])
        if product is None or product["description"] != change["before"]:
            raise ValueError("Proposed original description does not match the catalog")
        product["description"] = change["after"]
    return proposal, catalog, proposal_digest(proposal)


def preview_text(catalog_path: Path, proposal_path: Path) -> str:
    proposal, updated, reviewed_sha = preview(catalog_path, proposal_path)
    original = deepcopy(updated)
    originals = {item["sku"]: item for item in original["products"]}
    for change in proposal["changes"]:
        originals[change["sku"]]["description"] = change["before"]
    before = json.dumps(original, indent=2, ensure_ascii=False).splitlines()
    after = json.dumps(updated, indent=2, ensure_ascii=False).splitlines()
    diff = "\n".join(difflib.unified_diff(
        before, after, fromfile="catalog (current)",
        tofile="catalog (proposed)", lineterm=""))
    reasons = "\n".join(f"{c['sku']}: {json.dumps(c['reason'], ensure_ascii=False)}"
                        for c in proposal["changes"])
    return (f"Proposal: {proposal['proposal_id']}\n"
            f"Generation: {proposal['generation']['mode']}\n"
            f"Source SHA-256: {proposal['source_sha256']}\n"
            f"Reviewed SHA-256: {reviewed_sha}\n\n{diff}\n\nReasons:\n{reasons}\n")


def decide(proposal_path: Path, decision_path: Path, *, choice: str,
           reviewer: str, reviewed_sha256: str) -> dict:
    """Record an explicit decision about the exact previewed proposal."""
    _separate(decision_path, proposal_path)
    proposal = read_proposal(proposal_path)
    if choice not in ("approve", "reject"):
        raise ValueError("Choice must be approve or reject")
    _text(reviewer, "reviewer", 100)
    if reviewed_sha256 != proposal_digest(proposal):
        raise ValueError("Proposal differs from the reviewed version; preview it again")
    decision = {
        "schema_version": 1, "proposal_id": proposal["proposal_id"],
        "proposal_sha256": reviewed_sha256, "choice": choice,
        "reviewer": reviewer,
        "recorded_at": datetime.now(timezone.utc).isoformat(),
    }
    write_new_json(decision_path, decision)
    return decision


def apply_reviewed(catalog_path: Path, proposal_path: Path,
                   decision_path: Path, output_path: Path) -> dict:
    """Write a separate catalog only after all revision and decision checks."""
    _separate(output_path, catalog_path, proposal_path, decision_path)
    proposal, updated, current_sha = preview(catalog_path, proposal_path)
    decision = parse_json(decision_path.read_text(encoding="utf-8"))
    _keys(decision, {"schema_version", "proposal_id", "proposal_sha256",
                     "choice", "reviewer", "recorded_at"}, "Decision")
    _schema(decision["schema_version"])
    _text(decision["reviewer"], "reviewer", 100)
    _text(decision["recorded_at"], "recorded_at", 100)
    if decision["choice"] != "approve":
        raise ValueError("This proposal is not approved")
    if (decision["proposal_id"] != proposal["proposal_id"]
            or decision["proposal_sha256"] != current_sha):
        raise ValueError("Decision does not cover the current proposal")
    write_new_json(output_path, updated)
    return updated


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    proposal = commands.add_parser("propose", help="Generate a pending fixture proposal")
    proposal.add_argument("--catalog", type=Path, required=True)
    proposal.add_argument("--response", type=Path, required=True)
    proposal.add_argument("--proposal", type=Path, required=True)
    proposal.add_argument("--workspace", type=Path,
                          default=Path(".swarms-review-workspace"))
    view = commands.add_parser("preview", help="Show exact changes and review digest")
    view.add_argument("--catalog", type=Path, required=True)
    view.add_argument("--proposal", type=Path, required=True)
    decision = commands.add_parser("decide", help="Explicitly approve or reject a preview")
    decision.add_argument("--proposal", type=Path, required=True)
    decision.add_argument("--decision", type=Path, required=True)
    decision.add_argument("--choice", choices=("approve", "reject"), required=True)
    decision.add_argument("--reviewer", required=True)
    decision.add_argument("--reviewed-sha256", required=True)
    apply = commands.add_parser("apply", help="Write an approved separate catalog")
    apply.add_argument("--catalog", type=Path, required=True)
    apply.add_argument("--proposal", type=Path, required=True)
    apply.add_argument("--decision", type=Path, required=True)
    apply.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "propose":
            backend = ScriptedModel(args.response.read_text(encoding="utf-8"))
            result = propose(args.catalog, args.proposal, llm=backend,
                             mode="scripted", workspace=args.workspace)
            print(f"Pending proposal: {result['proposal_id']} ({args.proposal})")
        elif args.command == "preview":
            print(preview_text(args.catalog, args.proposal), end="")
        elif args.command == "decide":
            decide(args.proposal, args.decision, choice=args.choice,
                   reviewer=args.reviewer, reviewed_sha256=args.reviewed_sha256)
            print(f"Decision recorded: {args.choice} ({args.decision})")
        else:
            apply_reviewed(args.catalog, args.proposal, args.decision, args.output)
            print(f"Reviewed catalog written: {args.output}")
        return 0
    except (ValueError, OSError) as error:
        print(f"Review workflow error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
