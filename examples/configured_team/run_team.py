"""Validate an authored YAML team, run that exact configuration, and retain its snapshot."""

import argparse
from copy import deepcopy
import hashlib
from importlib.metadata import version
import json
import os
from pathlib import Path
import platform
import sys

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator
import yaml


FRAMEWORK_VERSION = "15.0.3"
HERE = Path(__file__).parent
AGENT_POLICY = {
    "max_loops": 1, "retry_attempts": 1, "output_type": "final",
    "autosave": False, "persistent_memory": False, "context_compression": False,
    "dynamic_context_window": False, "streaming_on": False, "stream": False,
    "print_on": False, "verbose": False,
}


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class AgentDefinition(StrictModel):
    agent_name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,31}$")
    system_prompt: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    temperature: float = Field(ge=0, le=2, allow_inf_nan=False)

    @field_validator("system_prompt", "model_name")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Value must contain non-whitespace text")
        return value


class TeamConfig(StrictModel):
    schema_version: int
    name: str = Field(pattern=r"^[A-Za-z][A-Za-z0-9_-]{0,63}$")
    task: str = Field(min_length=1)
    agents: list[AgentDefinition] = Field(min_length=2)

    @field_validator("schema_version")
    @classmethod
    def known_version(cls, value):
        if value != 1:
            raise ValueError("This companion supports schema_version 1")
        return value

    @field_validator("task")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Task must contain non-whitespace text")
        return value

    @model_validator(mode="after")
    def distinct_names(self):
        names = [agent.agent_name for agent in self.agents]
        if len(names) != len(set(names)):
            raise ValueError("Agent names must be unique")
        return self


class UniqueKeyLoader(yaml.SafeLoader):
    pass


def unique_mapping(loader, node, deep=False):
    result = {}
    for key_node, value_node in node.value:
        key = loader.construct_object(key_node, deep=deep)
        if not isinstance(key, str):
            raise ValueError("Configuration mapping keys must be strings")
        if key in result:
            raise ValueError(f"Duplicate YAML key {key!r} at line {key_node.start_mark.line + 1}")
        result[key] = loader.construct_object(value_node, deep=deep)
    return result


UniqueKeyLoader.add_constructor(yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG, unique_mapping)


def read_config(text: str) -> TeamConfig:
    """Validate before importing Swarms or constructing any Agent."""
    try:
        return TeamConfig.model_validate(yaml.load(text, Loader=UniqueKeyLoader))
    except yaml.YAMLError as error:
        raise ValueError(f"Invalid YAML: {error}") from error


def fingerprint(value) -> str:
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                            separators=(",", ":"))
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def native_config(config: TeamConfig) -> dict:
    return {"agents": [{**agent.model_dump(), **AGENT_POLICY} for agent in config.agents]}


class ScriptedModel:
    def __init__(self, response):
        self.response = response
        self.messages = []

    def run(self, task=None, messages=None, **kwargs):
        self.messages.append(deepcopy(messages))
        return self.response


def validate_responses(config, responses):
    if not isinstance(responses, dict) or set(responses) != {a.agent_name for a in config.agents}:
        raise ValueError("Fixtures must supply exactly one response for each configured agent name")
    if any(not isinstance(value, str) for value in responses.values()):
        raise ValueError("Each scripted response must be text")


def build_team(config: TeamConfig, *, workspace: Path, responses: dict | None):
    """Use unchanged YAML loading; replace model backends only in scripted mode."""
    if responses is not None:
        validate_responses(config, responses)
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    if version("swarms") != FRAMEWORK_VERSION:
        raise ValueError(f"Install swarms=={FRAMEWORK_VERSION} for this companion")
    from swarms import AgentLoader

    payload = yaml.safe_dump(native_config(config), allow_unicode=True, sort_keys=False)
    _, agents = AgentLoader().load_agents_from_yaml(
        yaml_file=None, yaml_string=payload, return_type="both",
    )
    if [a.agent_name for a in agents] != [a.agent_name for a in config.agents]:
        raise RuntimeError("Loaded agent order differs from the validated configuration")
    if responses is not None:
        for agent in agents:
            agent.llm = ScriptedModel(responses[agent.agent_name])
    return agents


def run_config(config: TeamConfig, *, workspace: Path, responses: dict | None) -> dict:
    agents = build_team(config, workspace=workspace, responses=responses)
    from swarms import SequentialWorkflow

    workflow = SequentialWorkflow(name=config.name, agents=agents, max_loops=1,
                                  output_type="dict", autosave=False, team_awareness=False,
                                  multi_agent_collab_prompt=False, drift_detection=False)
    starts = [len(agent.short_memory.conversation_history) for agent in agents]
    failure = None
    try:
        workflow.run(config.task)
    except Exception as error:
        failure = type(error).__name__
    stages = []
    for agent, start in zip(agents, starts):
        replies = [row.get("content") for row in agent.short_memory.conversation_history[start:]
                   if row.get("role") == agent.agent_name]
        output = replies[-1] if replies else None
        usable = isinstance(output, str) and bool(output.strip())
        stages.append({"agent_name": agent.agent_name, "status": "responded" if usable else "no_response",
                       "output": output if isinstance(output, str) else None})
    complete = failure is None and all(stage["status"] == "responded" for stage in stages)
    snapshot = config.model_dump()
    native = native_config(config)
    return {
        "report_version": 1, "status": "complete" if complete else "failed",
        "mode": "scripted" if responses is not None else "provider",
        "notice": "Scripted responses exercise configuration and handoffs; they do not perform the task."
        if responses is not None else "Provider output can differ between runs of the same configuration.",
        "config": snapshot, "config_sha256": fingerprint(snapshot),
        "native_agents": native, "native_agents_sha256": fingerprint(native),
        "runtime": {"swarms": version("swarms"), "python": platform.python_version(),
                    "PyYAML": version("PyYAML")},
        "workflow": {"type": "SequentialWorkflow", "max_loops": 1, "agent_order": [a.agent_name for a in agents]},
        "scripted_responses": responses, "stages": stages, "workflow_error": failure,
        "final_output": stages[-1]["output"] if complete else None,
    }


def replay_config(report: dict) -> TeamConfig:
    """Check the stored configuration matches its declared identity and protocol."""
    try:
        if type(report["report_version"]) is not int or report["report_version"] != 1:
            raise ValueError("Unknown report version")
        if report["runtime"]["swarms"] != FRAMEWORK_VERSION:
            raise ValueError("Report framework version differs from this companion")
        config = TeamConfig.model_validate(report["config"])
        if fingerprint(config.model_dump()) != report["config_sha256"]:
            raise ValueError("Stored configuration does not match its digest")
        native = native_config(config)
        if native != report["native_agents"] or fingerprint(native) != report["native_agents_sha256"]:
            raise ValueError("Stored native configuration differs from this companion's execution policy")
        return config
    except (KeyError, TypeError) as error:
        raise ValueError("Incomplete configuration report") from error


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    validate = actions.add_parser("validate")
    validate.add_argument("config", type=Path)
    run = actions.add_parser("run")
    run.add_argument("config", type=Path)
    replay = actions.add_parser("replay")
    replay.add_argument("report", type=Path)
    for action in (run, replay):
        action.add_argument("--output", type=Path, required=True)
        action.add_argument("--workspace", type=Path, default=Path("agent_workspace_configured"))
    choice = run.add_mutually_exclusive_group()
    choice.add_argument("--responses", type=Path, default=HERE / "responses.json")
    choice.add_argument("--live", action="store_true", help="Opt into provider calls and their costs")
    args = parser.parse_args(argv)
    try:
        if args.action in {"validate", "run"}:
            config = read_config(args.config.read_text(encoding="utf-8"))
        else:
            previous = json.loads(args.report.read_text(encoding="utf-8"))
            config = replay_config(previous)
            if previous.get("mode") != "scripted":
                raise ValueError("Replay command accepts scripted reports only; use run --live to opt into inference")
        if args.action == "validate":
            print(json.dumps({"valid": True, "config_sha256": fingerprint(config.model_dump()),
                              "agent_order": [a.agent_name for a in config.agents]}, indent=2))
            return 0
        responses = (previous["scripted_responses"] if args.action == "replay" else
                     None if args.live else json.loads(args.responses.read_text(encoding="utf-8")))
        if args.action == "replay" and responses is None:
            raise ValueError("Scripted replay requires stored response fixtures")
        report = run_config(config, workspace=args.workspace, responses=responses)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "mode": report["mode"], "output": str(args.output)}, indent=2))
        return 0 if report["status"] == "complete" else 1
    except (ValueError, OSError, KeyError) as error:
        print(f"Configuration run stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
