"""Exercise real registry/agents through FastAPI's in-process HTTP client."""

from concurrent.futures import ThreadPoolExecutor
import json
import os
from pathlib import Path
import subprocess
import sys
import threading

from fastapi.testclient import TestClient
import httpx
import pytest

from service import JobService, ScriptedModel, create_app, make_agent
from client import PollingExpired, poll


def settle(service, task_id):
    future = service.registry.get_task(task_id).future
    try:
        future.result(timeout=10)
    except Exception:
        # A failed/cancelled job is a valid terminal state; a timeout is not.
        assert future.done(), "Job did not settle"
    return service.get(task_id)


class BlockingModel:
    def __init__(self):
        self.started = threading.Event()
        self.release = threading.Event()
        self.messages = None

    def run(self, task=None, messages=None, **kwargs):
        self.messages = messages
        self.started.set()
        assert self.release.wait(10), "Test failed to release the model"
        return "Completed scripted job"


def test_submit_poll_result_and_unknown_id(tmp_path):
    service = JobService(lambda: make_agent(llm=ScriptedModel()), workspace=tmp_path)
    with TestClient(create_app(service)) as client:
        response = client.post("/jobs", json={"task": "Write a short project update"})
        assert response.status_code == 202
        task_id = response.json()["id"]
        assert response.headers["location"] == f"/jobs/{task_id}"
        assert settle(service, task_id)["status"] == "succeeded"
        result = client.get(response.json()["status_url"])
        assert result.status_code == 200
        assert result.json()["result"] == ScriptedModel().run()
        assert result.json()["error"] is None
        assert client.get("/jobs/missing").status_code == 404
        assert client.delete("/jobs/missing").status_code == 404


def test_queued_status_capacity_and_pending_cancellation(tmp_path):
    model = BlockingModel()
    built = []

    def factory():
        agent = make_agent(llm=model)
        built.append(agent)
        return agent

    service = JobService(factory, workspace=tmp_path, workers=1, max_active=2)
    with TestClient(create_app(service)) as client:
        try:
            first = client.post("/jobs", json={"task": "first"}).json()["id"]
            assert model.started.wait(5)
            second = client.post("/jobs", json={"task": "second"}).json()["id"]
            assert client.get(f"/jobs/{first}").json()["status"] == "running"
            assert client.get(f"/jobs/{second}").json()["status"] == "queued"
            # The released registry itself labels even a queued future RUNNING.
            assert service.registry.get_task(second).status.value == "running"
            rejected = client.post("/jobs", json={"task": "third"})
            assert rejected.status_code == 429
            assert rejected.headers["retry-after"] == "1"
            assert client.get("/health").json()["active"] == 2
            assert len(built) == 1  # Queued/rejected work has constructed no Agent.
            assert client.delete(f"/jobs/{first}").status_code == 409
            cancelled = client.delete(f"/jobs/{second}")
            assert cancelled.status_code == 200
            assert cancelled.json()["status"] == "cancelled"
            assert client.get("/health").json()["active"] == 1
            third = client.post("/jobs", json={"task": "third"})
            assert third.status_code == 202
            assert client.delete(third.json()["status_url"]).status_code == 200
        finally:
            model.release.set()
        assert settle(service, first)["status"] == "succeeded"


def test_concurrent_submitters_cannot_exceed_admission_limit(tmp_path):
    model = BlockingModel()
    service = JobService(lambda: make_agent(llm=model), workspace=tmp_path,
                         workers=1, max_active=1)
    with TestClient(create_app(service)) as client:
        try:
            with ThreadPoolExecutor(max_workers=8) as pool:
                responses = list(pool.map(lambda i: client.post("/jobs", json={"task": f"task {i}"}), range(8)))
            assert sum(r.status_code == 202 for r in responses) == 1
            assert sum(r.status_code == 429 for r in responses) == 7
            assert service.stats()["active"] == 1
        finally:
            model.release.set()
        accepted = next(r for r in responses if r.status_code == 202)
        settle(service, accepted.json()["id"])


def test_health_stays_responsive_during_agent_construction(tmp_path):
    constructing = threading.Event()
    release = threading.Event()

    def factory():
        constructing.set()
        assert release.wait(10)
        return make_agent(llm=ScriptedModel())

    service = JobService(factory, workspace=tmp_path, workers=1, max_active=1)
    with TestClient(create_app(service)) as client:
        try:
            submitted = client.post("/jobs", json={"task": "slow construction"})
            assert submitted.status_code == 202
            assert constructing.wait(5)
            assert client.get("/health").json()["status"] == "ok"
            assert service.get(submitted.json()["id"])["status"] == "running"
        finally:
            release.set()
        settle(service, submitted.json()["id"])


@pytest.mark.parametrize("failure", ["construction", "backend", "empty"])
def test_failures_are_terminal_and_release_capacity(tmp_path, failure):
    class FailingModel:
        def run(self, **kwargs):
            if failure == "backend":
                raise RuntimeError("Scripted provider failure")
            return ""

    def factory():
        if failure == "construction":
            raise ValueError("Scripted construction failure")
        return make_agent(llm=FailingModel())

    service = JobService(factory, workspace=tmp_path, workers=1, max_active=1)
    with TestClient(create_app(service)) as client:
        response = client.post("/jobs", json={"task": "A task that must not become its own answer"})
        job = settle(service, response.json()["id"])
        assert job["status"] == "failed"
        assert job["result"] is None
        assert job["error"]["code"] == "agent_failed"
        assert client.get("/health").json()["active"] == 0
        assert "Scripted" not in str(job["error"])


def test_each_job_has_its_own_real_agent_history(tmp_path):
    agents = []

    def factory():
        agent = make_agent(llm=ScriptedModel())
        agents.append(agent)
        return agent

    service = JobService(factory, workspace=tmp_path)
    with TestClient(create_app(service)) as client:
        ids = [client.post("/jobs", json={"task": f"private task {i}"}).json()["id"] for i in range(2)]
        for task_id in ids:
            settle(service, task_id)
        assert len(agents) == 2 and agents[0] is not agents[1]
        histories = [str(agent.short_memory.conversation_history) for agent in agents]
        assert sum("private task 0" in history for history in histories) == 1
        assert sum("private task 1" in history for history in histories) == 1
        assert not any("private task 0" in history and "private task 1" in history for history in histories)


def test_retained_record_limit_survives_completion(tmp_path):
    service = JobService(lambda: make_agent(llm=ScriptedModel()), workspace=tmp_path,
                         workers=1, max_active=1, max_records=1)
    with TestClient(create_app(service)) as client:
        task_id = client.post("/jobs", json={"task": "first"}).json()["id"]
        settle(service, task_id)
        assert service.stats()["active"] == 0
        rejected = client.post("/jobs", json={"task": "second"})
        assert rejected.status_code == 503
        assert service.stats()["retained"] == 1
        assert client.get("/health").json()["accepting"] is False
        assert client.get(f"/jobs/{task_id}").json()["status"] == "succeeded"


def test_shutdown_cancels_queue_but_cannot_stop_running_thread(tmp_path):
    model = BlockingModel()
    service = JobService(lambda: make_agent(llm=model), workspace=tmp_path,
                         workers=1, max_active=2)
    with TestClient(create_app(service, drain_seconds=0)) as client:
        try:
            first = client.post("/jobs", json={"task": "running"}).json()["id"]
            assert model.started.wait(5)
            second = client.post("/jobs", json={"task": "queued"}).json()["id"]
            assert service.close(drain_seconds=0) == 1
            assert service.get(first)["status"] == "running"
            assert service.get(second)["status"] == "cancelled"
            assert client.post("/jobs", json={"task": "late"}).status_code == 503
        finally:
            model.release.set()
        assert settle(service, first)["status"] == "succeeded"


@pytest.mark.parametrize("body", [{}, {"task": ""}, {"task": " \n"}, {"task": None},
                                    {"task": 42}, {"task": "x" * 8001}, {"task": "ok", "extra": 1}])
def test_invalid_requests_do_not_allocate_registry_records(tmp_path, body):
    service = JobService(lambda: pytest.fail("Unexpected Agent construction"), workspace=tmp_path)
    with TestClient(create_app(service)) as client:
        assert client.post("/jobs", json=body).status_code == 422
        assert service.stats()["retained"] == 0


def test_empty_service_shutdown_does_not_wait(tmp_path):
    service = JobService(lambda: pytest.fail("Unexpected construction"), workspace=tmp_path)
    assert service.close(0) == 0
    assert service.stats()["accepting"] is False


@pytest.mark.parametrize("seconds", [-1, float("nan"), float("inf"), True])
def test_invalid_shutdown_grace_does_not_close_service(tmp_path, seconds):
    service = JobService(lambda: pytest.fail("Unexpected construction"), workspace=tmp_path)
    try:
        with pytest.raises(ValueError):
            service.close(seconds)
        assert service.stats()["accepting"] is True
    finally:
        service.close(0)


def test_poll_timeout_does_not_cancel_or_resubmit_a_live_job(tmp_path):
    model = BlockingModel()
    service = JobService(lambda: make_agent(llm=model), workspace=tmp_path)
    with TestClient(create_app(service)) as client:
        try:
            task_id = client.post("/jobs", json={"task": "Wait for release"}).json()["id"]
            assert model.started.wait(5)
            with pytest.raises(PollingExpired, match=task_id):
                poll(client, task_id, wait_seconds=0)
            assert service.get(task_id)["status"] == "running"
            assert service.stats()["retained"] == 1
        finally:
            model.release.set()
        settle(service, task_id)
        assert poll(client, task_id, wait_seconds=0)["status"] == "succeeded"


def test_poll_does_not_start_another_request_at_the_deadline(monkeypatch):
    clock = [0.0]
    calls = []
    monkeypatch.setattr("client.time.monotonic", lambda: clock[0])
    monkeypatch.setattr("client.time.sleep", lambda duration: clock.__setitem__(0, clock[0] + duration))

    class PendingClient:
        def get(self, path):
            calls.append(clock[0])
            return httpx.Response(200, json={"status": "running"},
                                  request=httpx.Request("GET", "http://local" + path))

    with pytest.raises(PollingExpired):
        poll(PendingClient(), "task-example", wait_seconds=0.1)
    assert calls == [0.0]


@pytest.mark.parametrize("action,status,expected", [
    ("poll", "succeeded", 0), ("poll", "failed", 1),
    ("poll", "cancelled", 1), ("cancel", "cancelled", 0),
])
def test_terminal_cli_unicode_and_exit_codes(action, status, expected):
    # A subprocess exercises a real cp1252-configured redirected stdout stream.
    code = '''
import client, httpx, sys
class LocalClient:
    def __init__(self, **kwargs): pass
    def __enter__(self): return self
    def __exit__(self, *args): pass
    def get(self, path):
        return httpx.Response(200, json={"id":"task-example", "status":STATUS,
            "result":"Hello \\U0001f331" if STATUS == "succeeded" else None, "error":None},
            request=httpx.Request("GET", "http://local"+path))
    delete = get
client.httpx.Client = LocalClient
sys.argv = ["client.py", ACTION, "task-example"]
raise SystemExit(client.main())
'''.replace("STATUS", repr(status)).replace("ACTION", repr(action))
    environment = {**os.environ, "PYTHONIOENCODING": "cp1252"}
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent,
                            env=environment, capture_output=True, timeout=10)
    assert result.returncode == expected, result.stderr.decode("utf-8", errors="replace")
    report = json.loads(result.stdout.decode("utf-8"))
    assert report["status"] == status
    if status == "succeeded":
        assert report["result"] == "Hello 🌱"
