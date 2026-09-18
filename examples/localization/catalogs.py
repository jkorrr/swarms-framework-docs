"""Generate, validate, and render simple message catalogs with Swarms."""

import argparse
from collections import Counter
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import re
from string import Formatter
import sys
from typing import Any, Callable

LOCALE = re.compile(r"[a-z]{2}(?:-[A-Z]{2})?")
KEY = re.compile(r"[a-z][a-z0-9_.-]{0,63}")
FIELD = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")


def parse_json(text: str) -> Any:
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError("Duplicate JSON key")
            result[key] = value
        return result

    def invalid(value):
        raise ValueError("Non-finite JSON constant")

    return json.loads(text, object_pairs_hook=pairs, parse_constant=invalid)


def exact_keys(value: Any, keys: set[str], name: str) -> None:
    if not isinstance(value, dict) or set(value) != keys:
        raise ValueError(f"{name} has missing or unexpected fields")


def locale_name(value: Any) -> None:
    if not isinstance(value, str) or not LOCALE.fullmatch(value):
        raise ValueError("Use locale names such as en, es, fr, or pt-BR")


def placeholders(template: Any) -> Counter:
    """Accept named string fields and escaped braces; count repeated fields."""
    if not isinstance(template, str) or not template.strip() or len(template) > 1000:
        raise ValueError("Messages must contain 1 to 1000 characters")
    counts = Counter()
    for _, field, spec, conversion in Formatter().parse(template):
        if field is not None:
            if not FIELD.fullmatch(field) or spec or conversion:
                raise ValueError("Only simple named placeholders are supported")
            counts[field] += 1
    return counts


def validate_source(source: Any) -> None:
    exact_keys(source, {"schema_version", "source_locale", "messages"}, "Source")
    if type(source["schema_version"]) is not int or source["schema_version"] != 1:
        raise ValueError("Only schema_version=1 is supported")
    locale_name(source["source_locale"])
    messages = source["messages"]
    if not isinstance(messages, dict) or not 1 <= len(messages) <= 100:
        raise ValueError("Provide 1 to 100 source messages")
    for key, template in messages.items():
        if not isinstance(key, str) or not KEY.fullmatch(key):
            raise ValueError("Invalid message key")
        placeholders(template)


def validate_translation(source: dict, locale: str, candidate: Any) -> dict:
    locale_name(locale)
    exact_keys(candidate, {"locale", "messages"}, "Translation")
    if candidate["locale"] != locale:
        raise ValueError("Response locale does not match its assigned agent")
    messages = candidate["messages"]
    exact_keys(messages, set(source["messages"]), "Message catalog")
    for key, template in messages.items():
        if placeholders(template) != placeholders(source["messages"][key]):
            raise ValueError(f"Placeholder counts differ for message {key}")
    return deepcopy(messages)


def prepare_runtime(workspace: Path) -> None:
    """Call once before importing Swarms in a fresh process."""
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
    if "swarms" in sys.modules:
        from swarms.telemetry.otel import swarm_telemetry
        if swarm_telemetry().ready:
            raise ValueError("Restart with native Swarms telemetry disabled")


def _require_runtime() -> None:
    if (os.getenv("SWARMS_TELEMETRY_ON") != "false"
            or os.getenv("LITELLM_LOCAL_MODEL_COST_MAP") != "True"):
        raise ValueError("Call prepare_runtime before importing Swarms")
    from swarms.telemetry.otel import swarm_telemetry
    if swarm_telemetry().ready:
        raise ValueError("Restart with native Swarms telemetry disabled")


class ScriptedModel:
    """A fixed backend for contract tests, not a translation model."""

    def __init__(self, response: str, *, fail: bool = False):
        self.response = response
        self.fail = fail
        self.calls = []

    def run(self, task=None, messages=None, **kwargs):
        self.calls.append(deepcopy(messages))
        if self.fail:
            raise RuntimeError("Synthetic translation backend failure")
        return self.response


def make_agent(locale: str, llm: Any, *, model_name: str = "gpt-4o-mini") -> Any:
    locale_name(locale)
    _require_runtime()
    from swarms import Agent
    prompt = (
        f"Translate every message in the supplied source catalog into {locale}. "
        "Return only JSON with exactly locale and messages fields. Set locale "
        f"to {locale}. Preserve every message key and each named placeholder "
        "including its repetition count. Reorder placeholders if grammar needs "
        "it. Use double braces for literal braces. Do not add fields, explanations, "
        "ICU plural syntax, format specifications, or Markdown fences."
    )
    return Agent(
        agent_name=f"catalog-{locale}", system_prompt=prompt, llm=llm,
        model_name=model_name, max_loops=1, retry_attempts=1,
        output_type="final", print_on=False, autosave=False,
        persistent_memory=False, context_compression=False,
        dynamic_context_window=False, streaming_on=False, stream=False,
    )


def generate_catalogs(source: dict, locales: list[str], *,
                      agent_factory: Callable[[str], Any], mode: str = "custom") -> dict:
    validate_source(source)
    if (not isinstance(locales, list) or not 1 <= len(locales) <= 8
            or any(not isinstance(item, str) for item in locales)
            or len(set(locales)) != len(locales)):
        raise ValueError("Provide 1 to 8 distinct target locales")
    for locale in locales:
        locale_name(locale)
        if locale == source["source_locale"]:
            raise ValueError("Target locales must differ from the source locale")
    if mode not in {"scripted", "provider", "custom"}:
        raise ValueError("Unknown generation mode")
    _require_runtime()
    from swarms import ConcurrentWorkflow

    agents = [agent_factory(locale) for locale in locales]
    if len({id(agent) for agent in agents}) != len(agents):
        raise ValueError("Return a fresh Agent for each locale")
    for locale, agent in zip(locales, agents):
        if agent.agent_name != f"catalog-{locale}":
            raise ValueError("Agent role does not match its assigned locale")
    starts = [len(agent.short_memory.conversation_history) for agent in agents]
    workflow = ConcurrentWorkflow(
        name="message-localization", agents=agents, output_type="dict-all-except-first",
        auto_save=False, autosave=False, show_dashboard=False,
        on_error="store", max_workers=len(agents),
    )
    rows = workflow.run(json.dumps(source, ensure_ascii=False))
    if not isinstance(rows, list):
        raise ValueError("Unexpected workflow result format")
    expected_roles = {agent.agent_name for agent in agents}
    by_role = {}
    for row in rows:
        if (not isinstance(row, dict) or not isinstance(row.get("role"), str)
                or row["role"] not in expected_roles | {f"{r} (failed)" for r in expected_roles}):
            raise ValueError("Unexpected workflow result role")
        by_role.setdefault(row["role"], []).append(row)

    catalogs, results = {}, []
    for locale, agent, start in zip(locales, agents, starts):
        matches = by_role.get(agent.agent_name, [])
        replied = any(row.get("role") == agent.agent_name
                      for row in agent.short_memory.conversation_history[start:])
        if len(matches) != 1 or by_role.get(f"{agent.agent_name} (failed)") or not replied:
            results.append({"locale": locale, "status": "execution_error",
                            "detail": "No single successful assistant result"})
            continue
        try:
            output = matches[0]["content"]
            if not isinstance(output, str):
                raise ValueError("Response must be JSON text")
            candidate = parse_json(output)
            catalogs[locale] = validate_translation(source, locale, candidate)
        except (ValueError, KeyError, TypeError) as error:
            results.append({"locale": locale, "status": "invalid_catalog", "detail": str(error)})
        else:
            results.append({"locale": locale, "status": "accepted", "detail": "Contract passed"})
    return {"schema_version": 1, "mode": mode, "framework_version": version("swarms"),
            "source": deepcopy(source), "catalogs": catalogs, "results": results}


class CatalogStore:
    """Validate a bundle once, then render exact locale matches or source fallback."""

    def __init__(self, bundle: Any):
        if not isinstance(bundle, dict) or type(bundle.get("schema_version")) is not int or bundle["schema_version"] != 1:
            raise ValueError("Unsupported bundle")
        source = bundle.get("source")
        validate_source(source)
        catalogs = bundle.get("catalogs")
        if not isinstance(catalogs, dict) or len(catalogs) > 8:
            raise ValueError("Invalid catalog collection")
        self._source = deepcopy(source)
        self._catalogs = {}
        for locale, messages in catalogs.items():
            if locale == source["source_locale"]:
                raise ValueError("Source locale must not be duplicated")
            self._catalogs[locale] = validate_translation(
                source, locale, {"locale": locale, "messages": messages})

    def render(self, locale: str, key: str, values: dict[str, str]) -> dict:
        locale_name(locale)
        if not isinstance(key, str) or key not in self._source["messages"]:
            raise ValueError("Unknown message key")
        needed = set(placeholders(self._source["messages"][key]))
        if (not isinstance(values, dict) or set(values) != needed
                or any(not isinstance(value, str) for value in values.values())):
            raise ValueError("Supply exactly the required named string values")
        used = locale if locale in self._catalogs else self._source["source_locale"]
        messages = self._catalogs.get(locale, self._source["messages"])
        return {"text": messages[key].format_map(values), "requested_locale": locale,
                "used_locale": used, "fallback": used != locale}


def main() -> int:
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate")
    generate.add_argument("--source", type=Path, required=True)
    generate.add_argument("--locales", nargs="+", required=True)
    backend = generate.add_mutually_exclusive_group(required=True)
    backend.add_argument("--fixtures", type=Path)
    backend.add_argument("--model")
    generate.add_argument("--output", type=Path, required=True)
    generate.add_argument("--workspace", type=Path, default=Path("localization-workspace"))
    render = commands.add_parser("render")
    render.add_argument("bundle", type=Path)
    render.add_argument("--locale", required=True)
    render.add_argument("--key", required=True)
    render.add_argument("--value", action="append", default=[])
    args = parser.parse_args()
    try:
        if args.command == "render":
            store = CatalogStore(parse_json(args.bundle.read_text(encoding="utf-8")))
            values = {}
            for item in args.value:
                name, separator, value = item.partition("=")
                if not separator or name in values:
                    raise ValueError("Each --value must be a unique name=text pair")
                values[name] = value
            print(json.dumps(store.render(args.locale, args.key, values), ensure_ascii=False))
            return 0
        if args.output.exists():
            raise ValueError("Output exists; choose a new bundle path")
        source = parse_json(args.source.read_text(encoding="utf-8"))
        if args.fixtures:
            fixtures = parse_json(args.fixtures.read_text(encoding="utf-8"))
            if not isinstance(fixtures, dict) or any(locale not in fixtures for locale in args.locales):
                raise ValueError("Fixtures must contain every requested locale")
            agent_factory = lambda locale: make_agent(locale, ScriptedModel(
                json.dumps(fixtures[locale], ensure_ascii=False)))
            mode = "scripted"
        else:
            agent_factory = lambda locale: make_agent(locale, None, model_name=args.model)
            mode = "provider"
        prepare_runtime(args.workspace)
        bundle = generate_catalogs(source, args.locales, agent_factory=agent_factory, mode=mode)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(bundle, ensure_ascii=False, indent=2) + "\n")
        print(json.dumps(bundle["results"], ensure_ascii=False, indent=2))
        return 0 if all(row["status"] == "accepted" for row in bundle["results"]) else 1
    except (OSError, ValueError, TypeError, KeyError) as error:
        print(f"Catalog error: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
