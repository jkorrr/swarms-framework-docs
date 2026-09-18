"""A single-process, local Swarms job API with bounded admission."""

import argparse
import asyncio
from concurrent.futures import wait
from contextlib import asynccontextmanager
import os
import math
from pathlib import Path
import threading
from typing import Any, Callable
import uuid

from fastapi import FastAPI, HTTPException, Response
from pydantic import BaseModel, ConfigDict, Field, field_validator


class CapacityReached(Exception):
    pass


class ServiceClosed(Exception):
    pass


class NoAgentResponse(RuntimeError):
    pass


class JobRequest(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    task: str = Field(min_length=1, max_length=8_000)

    @field_validator("task")
    @classmethod
    def reject_blank(cls, task):
        if not task.strip():
            raise ValueError("Task must contain non-whitespace text")
        return task  # Preserve formatting inside the accepted task.


class ScriptedModel:
    """Fixed demonstration response; no model inference or task reasoning."""
    def run(self, task=None, messages=None, **kwargs):
        return "This is a scripted response from a real Swarms Agent invocation."


def make_agent(*, llm=None, model_name="gpt-4o-mini"):
    from swarms import Agent

    return Agent(
        agent_name=f"job-agent-{uuid.uuid4().hex}",
        system_prompt="Answer the user's task concisely. You have no tools.",
        model_name=model_name, llm=llm,
        max_loops=1, retry_attempts=1, output_type="final",
        autosave=False, persistent_memory=False, context_compression=False,
        dynamic_context_window=False, streaming_on=False, stream=False, print_on=False,
    )


class JobAgent:
    """Defer real Agent construction until a registry worker starts the job."""
    def __init__(self, factory: Callable[[], Any]):
        self.agent_name = f"job-worker-{uuid.uuid4().hex}"
        self.factory = factory

    def run(self, task: str) -> str:
        agent = self.factory()
        from swarms import Agent

        if not isinstance(agent, Agent):
            raise TypeError("The factory must return a Swarms Agent")
        start = len(agent.short_memory.conversation_history)
        result = agent.run(task=task)
        replies = [row for row in agent.short_memory.conversation_history[start:]
                   if row.get("role") == agent.agent_name]
        if not replies or not isinstance(result, str) or not result.strip():
            raise NoAgentResponse("The invocation did not produce a nonempty model response")
        return result


class JobService:
    """Use the public SubagentRegistry API; keep HTTP policy in the application."""
    def __init__(self, factory, *, workspace: Path, workers=2, max_active=4, max_records=100):
        if any(type(n) is not int or n < 1 for n in (workers, max_active, max_records)):
            raise ValueError("workers, max_active and max_records must be positive integers")
        if workers > max_active:
            raise ValueError("workers must not exceed max_active")
        workspace.mkdir(parents=True, exist_ok=True)
        os.environ["WORKSPACE_DIR"] = str(workspace.resolve())
        os.environ["LITELLM_LOCAL_MODEL_COST_MAP"] = "True"
        os.environ["SWARMS_TELEMETRY_ON"] = "false"
        from swarms import SubagentRegistry

        self.registry = SubagentRegistry(max_workers=workers, max_depth=0)
        self.factory = factory
        self.max_active = max_active
        self.max_records = max_records
        self._ids = set()
        self._closing = False
        self._lock = threading.Lock()

    def _active_count(self):
        return sum(not self.registry.get_task(task_id).future.done() for task_id in self._ids)

    def stats(self):
        with self._lock:
            active = self._active_count()
            return {"accepting": not self._closing and active < self.max_active
                    and len(self._ids) < self.max_records, "active": active,
                    "retained": len(self._ids), "max_active": self.max_active,
                    "max_records": self.max_records}

    def submit(self, task: str) -> str:
        # JobRequest also validates non-HTTP callers of this small service.
        task = JobRequest(task=task).task
        with self._lock:
            if self._closing:
                raise ServiceClosed("Service is shutting down")
            if self._active_count() >= self.max_active:
                raise CapacityReached("Active job capacity reached; poll existing jobs")
            if len(self._ids) >= self.max_records:
                raise ServiceClosed("Retained job limit reached; export results before restarting")
            task_id = self.registry.spawn(JobAgent(self.factory), task,
                                          max_retries=0, fail_fast=True)
            self._ids.add(task_id)
            return task_id

    def _task(self, task_id):
        if task_id not in self._ids:
            raise KeyError(task_id)
        return self.registry.get_task(task_id)

    def get(self, task_id: str) -> dict:
        with self._lock:
            future = self._task(task_id).future
            result = {"id": task_id, "status": None, "result": None, "error": None}
            # Registry 15.0.3 sets TaskStatus.RUNNING before submitting to the
            # executor. Future state distinguishes queueing from actual execution.
            if future.cancelled():
                result["status"] = "cancelled"
            elif future.running():
                result["status"] = "running"
            elif not future.done():
                result["status"] = "queued"
            elif future.cancelled():
                result["status"] = "cancelled"
            elif future.exception() is not None:
                result.update(status="failed", error={"code": "agent_failed",
                                                      "type": type(future.exception()).__name__})
            else:
                result.update(status="succeeded", result=future.result())
            return result

    def cancel(self, task_id: str) -> bool:
        with self._lock:
            self._task(task_id)
            return self.registry.cancel(task_id)

    def close(self, drain_seconds=5.0) -> int:
        """Reject new jobs, cancel queued jobs, then wait briefly for active ones.

        Return the count still running. This does not terminate Python threads.
        """
        if type(drain_seconds) not in (int, float) or not math.isfinite(drain_seconds) or drain_seconds < 0:
            raise ValueError("drain_seconds must be finite and nonnegative")
        with self._lock:
            self._closing = True
            futures = [self.registry.get_task(task_id).future for task_id in self._ids]
            for task_id in self._ids:
                self.registry.cancel(task_id)
            self.registry.shutdown()
        running = [future for future in futures if not future.done()]
        if not running:
            return 0
        _, pending = wait(running, timeout=drain_seconds)
        return sum(not future.done() for future in pending)


def create_app(service: JobService, *, drain_seconds=5.0) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app):
        yield
        remaining = await asyncio.to_thread(service.close, drain_seconds)
        if remaining:
            print(f"Shutdown grace elapsed: {remaining} job(s) still running; Python threads were not stopped.")

    app = FastAPI(title="Local Swarms background jobs", lifespan=lifespan)

    @app.get("/health")
    async def health():
        return {"status": "ok", **service.stats()}

    @app.post("/jobs", status_code=202)
    async def submit(request: JobRequest, response: Response):
        try:
            task_id = service.submit(request.task)
        except CapacityReached as exc:
            raise HTTPException(429, str(exc), headers={"Retry-After": "1"}) from exc
        except ServiceClosed as exc:
            raise HTTPException(503, str(exc)) from exc
        response.headers["Location"] = f"/jobs/{task_id}"
        return {"id": task_id, "status_url": f"/jobs/{task_id}"}

    @app.get("/jobs/{task_id}")
    async def get_job(task_id: str):
        try:
            return service.get(task_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown job ID") from exc

    @app.delete("/jobs/{task_id}")
    async def cancel_job(task_id: str):
        try:
            cancelled = service.cancel(task_id)
        except KeyError as exc:
            raise HTTPException(404, "Unknown job ID") from exc
        if not cancelled:
            raise HTTPException(409, "Job already started or finished; running threads cannot be cancelled")
        return service.get(task_id)

    return app


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--scripted", action="store_true", help="Fixed response fixture; no model inference")
    mode.add_argument("--model", help="Explicit provider model; may incur API charges")
    parser.add_argument("--port", type=int, default=8030)
    parser.add_argument("--workspace", type=Path, default=Path("agent_workspace_jobs"))
    args = parser.parse_args()
    factory = ((lambda: make_agent(llm=ScriptedModel())) if args.scripted
               else (lambda: make_agent(model_name=args.model)))
    service = JobService(factory, workspace=args.workspace)
    import uvicorn

    print("Local single-process demo; job records are lost when this process exits.")
    uvicorn.run(create_app(service), host="127.0.0.1", port=args.port, workers=1)


if __name__ == "__main__":
    main()
