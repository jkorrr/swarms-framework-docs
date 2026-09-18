"""Build a local image caption manifest, then export its usable drafts."""

import argparse
import base64
import csv
from hashlib import sha256
from importlib.metadata import version
from io import BytesIO
import json
import os
from pathlib import Path
import sys
import warnings

from PIL import Image, UnidentifiedImageError
from pydantic import BaseModel, ConfigDict, Field, field_validator

FRAMEWORK_VERSION = "15.0.3"
MAX_BYTES = 10 * 1024 * 1024
MAX_PIXELS = 20_000_000
FORMATS = {"PNG": "image/png", "JPEG": "image/jpeg", "WEBP": "image/webp"}
SUFFIXES = {".png", ".jpg", ".jpeg", ".webp"}
PROMPT = 'Describe the visible image in one concise caption. Return only JSON: {"caption": "..."}.'


class Caption(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    caption: str = Field(min_length=1, max_length=240)

    @field_validator("caption")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("Caption is blank")
        return value


class ScriptedModel:
    """A fixed response; it exercises the image plumbing, not visual inference."""

    def __init__(self, response):
        self.response = response

    def run(self, task=None, img=None, **kwargs):
        if not isinstance(img, str) or not img.startswith("data:image/"):
            raise ValueError("The fixture expected a per-image data URI")
        return self.response


def inspect_image(path):
    """Decode the exact bytes that will be passed to the Agent."""
    if path.stat().st_size > MAX_BYTES:
        raise ValueError("Image exceeds the 10 MiB input limit")
    data = path.read_bytes()
    if len(data) > MAX_BYTES:
        raise ValueError("Image exceeds the 10 MiB input limit")
    with warnings.catch_warnings():
        warnings.simplefilter("error", Image.DecompressionBombWarning)
        with Image.open(BytesIO(data)) as opened:
            if opened.format not in FORMATS:
                raise ValueError("Image bytes are not PNG, JPEG, or WebP")
            if getattr(opened, "n_frames", 1) != 1:
                raise ValueError("Use a single-frame image")
            width, height = opened.size
            if width * height > MAX_PIXELS:
                raise ValueError("Image exceeds the 20 megapixel input limit")
            mime = FORMATS[opened.format]
            opened.verify()
        with Image.open(BytesIO(data)) as opened:
            opened.load()
    return {"sha256": sha256(data).hexdigest(), "width": width, "height": height}, (
        f"data:{mime};base64," + base64.b64encode(data).decode("ascii")
    )


def make_agent(name, response, model):
    from swarms import Agent

    return Agent(
        agent_name=name, model_name=model, llm=ScriptedModel(response) if response is not None else None,
        system_prompt="Write a draft caption for the supplied image. Do not invent context outside the image.",
        max_loops=1, retry_attempts=1, output_type="final", multi_modal=True,
        autosave=False, persistent_memory=False, context_compression=False,
        dynamic_temperature_enabled=False, dynamic_context_window=False,
        print_on=False, verbose=False, streaming_on=False, stream=False,
    )


class ImageTask:
    """Bind one image to one Agent for the SDK's task-only batch helper."""

    def __init__(self, agent, image):
        self.agent = agent
        self.image = image

    def run(self, task):
        start = len(self.agent.short_memory.conversation_history)
        # 15.0.3's LiteLLM path prefers typed messages over the separate img.
        image_turn = {"role": "user", "content": [
            {"type": "text", "text": "Image to caption:"},
            {"type": "image_url", "image_url": {"url": self.image}},
        ]}
        raw = self.agent.run(task=task, img=self.image, messages=[image_turn])
        replies = [row.get("content") for row in self.agent.short_memory.conversation_history[start:]
                   if row.get("role") == self.agent.agent_name]
        if not replies or not isinstance(raw, str) or replies[-1] != raw:
            raise RuntimeError("Agent produced no new final assistant response")
        return raw


def build_catalog(folder, *, responses, workspace, batch_size=2, model="openai/gpt-4o", agent_factory=make_agent):
    if type(batch_size) is not int or not 1 <= batch_size <= 8:
        raise ValueError("batch_size must be an integer from 1 to 8")
    if not folder.is_dir():
        raise ValueError(f"Image folder does not exist: {folder}")
    paths = sorted((path for path in folder.iterdir() if path.is_file()), key=lambda path: path.name)
    if not paths:
        raise ValueError("Image folder has no files")
    if responses is not None and (not isinstance(responses, dict) or
                                  any(not isinstance(key, str) or not isinstance(value, str)
                                      for key, value in responses.items())):
        raise ValueError("Responses must map filenames to JSON response strings")
    os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
    os.environ["SWARMS_TELEMETRY_ON"] = "false"
    os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
    if version("swarms") != FRAMEWORK_VERSION:
        raise ValueError(f"Install swarms=={FRAMEWORK_VERSION}")
    from swarms.structs.multi_agent_exec import run_agents_with_different_tasks

    entries = []
    for offset in range(0, len(paths), batch_size):
        pairs, accepted = [], []
        for path in paths[offset:offset + batch_size]:
            entry = {"file": path.name, "status": "skipped", "reason": "Unsupported filename extension"}
            entries.append(entry)
            if path.suffix.lower() not in SUFFIXES:
                continue
            try:
                metadata, image = inspect_image(path)
            except (OSError, ValueError, UnidentifiedImageError,
                    Image.DecompressionBombError, Image.DecompressionBombWarning) as error:
                entry.update(status="invalid_image", reason=str(error))
                continue
            entry.update(metadata)
            if responses is not None and path.name not in responses:
                raise ValueError(f"Missing scripted response for {path.name}")
            response = None if responses is None else responses[path.name]
            agent = agent_factory(f"Caption-{len(entries)}", response, model)
            pairs.append((ImageTask(agent, image), PROMPT))
            accepted.append(entry)
        outputs = run_agents_with_different_tasks(pairs, batch_size=batch_size, max_workers=batch_size)
        for entry, raw in zip(accepted, outputs):
            if isinstance(raw, Exception):
                entry.update(status="model_error", reason=str(raw))
                continue
            entry["raw_output"] = raw
            try:
                caption = Caption.model_validate_json(raw).caption
            except ValueError as error:
                entry.update(status="invalid_caption", reason=str(error))
            else:
                entry.update(status="captioned", caption=caption)
                entry.pop("reason", None)
    failures = sum(row["status"] in {"invalid_image", "model_error", "invalid_caption"} for row in entries)
    return {"manifest_version": 1, "mode": "scripted" if responses is not None else "provider",
            "swarms": FRAMEWORK_VERSION, "model": model, "source_folder": str(folder.resolve()),
            "status": "partial" if failures else "complete", "entries": entries}


def export_csv(report, output, *, allow_partial=False):
    """Export only shape-validated draft captions; no inference or file rereads."""
    if (not isinstance(report, dict) or type(report.get("manifest_version")) is not int
            or report["manifest_version"] != 1 or not isinstance(report.get("mode"), str)
            or report["mode"] not in {"scripted", "provider"}
            or not isinstance(report.get("entries"), list)):
        raise ValueError("Expected a version 1 image catalog")
    rows, seen, failures = [], set(), False
    for entry in report["entries"]:
        if not isinstance(entry, dict):
            raise ValueError("Invalid manifest entry")
        name, status = entry.get("file"), entry.get("status")
        if not isinstance(name, str) or not name or name in seen:
            raise ValueError("Manifest filenames must be nonempty and unique")
        seen.add(name)
        if not isinstance(status, str) or status not in {"captioned", "skipped", "invalid_image", "model_error", "invalid_caption"}:
            raise ValueError(f"Unknown status for {name}")
        failures |= status in {"invalid_image", "model_error", "invalid_caption"}
        if status != "captioned":
            continue
        caption = Caption.model_validate({"caption": entry.get("caption")}).caption
        digest = entry.get("sha256")
        if (not isinstance(digest, str) or len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest)
                or any(type(entry.get(key)) is not int or entry[key] < 1 for key in ("width", "height"))):
            raise ValueError(f"Missing image identity or dimensions for {name}")
        rows.append({key: entry[key] for key in ("file", "sha256", "width", "height")} | {"caption": caption})
    if report.get("status") != ("partial" if failures else "complete"):
        raise ValueError("Manifest summary disagrees with its entries")
    if failures and not allow_partial:
        raise ValueError("Catalog is partial; inspect its entries or explicitly use --allow-partial")
    if not rows:
        raise ValueError("Catalog has no usable captions")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file", "sha256", "width", "height", "caption"])
        writer.writeheader()
        writer.writerows(rows)
    return len(rows)


def make_demo(folder):
    folder.mkdir(parents=True, exist_ok=False)
    images = folder / "images"
    images.mkdir()
    for name, color in (("red.png", "red"), ("blue-🌱.png", "blue")):
        Image.new("RGB", (32, 32), color).save(images / name)
    responses = {"red.png": '{"caption": "A red square."}',
                 "blue-🌱.png": '{"caption": "A blue square, café example 🌱."}'}
    (folder / "responses.json").write_text(json.dumps(responses, ensure_ascii=False, indent=2), encoding="utf-8")


def main(argv=None):
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    actions = parser.add_subparsers(dest="action", required=True)
    demo = actions.add_parser("demo")
    demo.add_argument("folder", type=Path)
    build = actions.add_parser("build")
    build.add_argument("folder", type=Path)
    build.add_argument("--output", type=Path, required=True)
    mode = build.add_mutually_exclusive_group(required=True)
    mode.add_argument("--responses", type=Path)
    mode.add_argument("--live", action="store_true", help="Opt into provider calls and costs")
    build.add_argument("--model", default="openai/gpt-4o")
    build.add_argument("--batch-size", type=int, default=2)
    build.add_argument("--workspace", type=Path, default=Path("agent_workspace_images"))
    export = actions.add_parser("export")
    export.add_argument("manifest", type=Path)
    export.add_argument("--output", type=Path, required=True)
    export.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    try:
        if args.action == "demo":
            make_demo(args.folder)
            print("Created demo images/ and responses.json")
            return 0
        if args.action == "export":
            if args.output.resolve() == args.manifest.resolve():
                raise ValueError("CSV output must differ from the manifest")
            count = export_csv(json.loads(args.manifest.read_text(encoding="utf-8")), args.output,
                               allow_partial=args.allow_partial)
            print(f"Exported {count} draft captions")
            return 0
        if args.output.resolve().is_relative_to(args.folder.resolve()):
            raise ValueError("Write the manifest outside the image folder")
        responses = None if args.live else json.loads(args.responses.read_text(encoding="utf-8"))
        if not args.live and not isinstance(responses, dict):
            raise ValueError("--responses must contain an object mapping filenames to JSON response strings")
        report = build_catalog(args.folder, responses=responses, workspace=args.workspace,
                               batch_size=args.batch_size, model=args.model)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"status": report["status"], "files": len(report["entries"])}))
        return 1 if report["status"] == "partial" else 0
    except (OSError, ValueError) as error:
        print(f"Catalog stopped: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
