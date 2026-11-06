"""Shared fakes for the GCP backend tests: in-memory Cloud Batch and Kubernetes API servers."""

from __future__ import annotations

import copy
import urllib.parse
from typing import Any

import pytest

from gcp_exec.submission import SchedulerApiError

WORKER_SA = "gx-batch-worker-sa@ut-test.iam.gserviceaccount.com"


class FakeBatchServer:
    """Minimal Cloud Batch v1 REST semantics: create (jobId/requestId), get, cancel.

    `fail_next_post_after_create` simulates a lost response: the job is created server-side,
    then the client sees a transport error (status None).
    """

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.request_ids: dict[str, str] = {}
        self.fail_next_post_after_create = False
        self.fail_posts_with: list[int | None] = []

    def __call__(self, method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        self.calls.append((method, url, copy.deepcopy(body)))
        path, _, query = url.partition("?")
        path = path.split("/v1/", 1)[1]
        params = dict(urllib.parse.parse_qsl(query))
        if method == "POST" and path.endswith(":cancel"):
            name = path[: -len(":cancel")]
            if name not in self.jobs:
                raise SchedulerApiError("not found", status_code=404)
            self.jobs[name]["status"]["state"] = "CANCELLATION_IN_PROGRESS"
            return {}
        if method == "POST":
            if self.fail_posts_with:
                raise SchedulerApiError("injected", status_code=self.fail_posts_with.pop(0))
            name = f"{path}/{params['jobId']}"
            rid = params.get("requestId")
            if name in self.jobs:
                if rid and self.request_ids.get(name) == rid:
                    return copy.deepcopy(self.jobs[name])  # requestId dedup within 60 min
                raise SchedulerApiError("ALREADY_EXISTS", status_code=409)
            job = copy.deepcopy(body or {})
            job["name"] = name
            job["createTime"] = "2026-11-06T08:00:00.123456789Z"
            job["status"] = {"state": "QUEUED", "statusEvents": []}
            self.jobs[name] = job
            if rid:
                self.request_ids[name] = rid
            if self.fail_next_post_after_create:
                self.fail_next_post_after_create = False
                raise SchedulerApiError("read timeout", status_code=None)
            return copy.deepcopy(job)
        if method == "GET":
            if path not in self.jobs:
                raise SchedulerApiError("not found", status_code=404)
            return copy.deepcopy(self.jobs[path])
        raise AssertionError(f"unexpected {method} {url}")

    def posts(self) -> list[tuple[str, str, dict[str, Any] | None]]:
        return [c for c in self.calls if c[0] == "POST" and ":cancel" not in c[1]]

    def set_state(self, name: str, state: str, **status: Any) -> None:
        self.jobs[name]["status"] = {"state": state, "statusEvents": [], **status}


class FakeK8sServer:
    """Minimal batch/v1 Job + core/v1 Pod list semantics."""

    def __init__(self) -> None:
        self.jobs: dict[str, dict[str, Any]] = {}
        self.pods: dict[str, list[dict[str, Any]]] = {}
        self.calls: list[tuple[str, str, dict[str, Any] | None]] = []
        self.fail_next_post_after_create = False

    def __call__(self, method: str, url: str, body: dict[str, Any] | None) -> dict[str, Any]:
        self.calls.append((method, url, copy.deepcopy(body)))
        path, _, query = url.partition("?")
        if "/api/v1/" in path and path.endswith("/pods"):
            selector = dict(urllib.parse.parse_qsl(query)).get("labelSelector", "")
            job = selector.split("=", 1)[1] if "=" in selector else ""
            return {"items": copy.deepcopy(self.pods.get(job, []))}
        tail = path.split("/jobs", 1)[1].strip("/")
        if method == "POST":
            name = body["metadata"]["name"]
            if name in self.jobs:
                raise SchedulerApiError("AlreadyExists", status_code=409)
            job = copy.deepcopy(body)
            job["status"] = {}
            self.jobs[name] = job
            if self.fail_next_post_after_create:
                self.fail_next_post_after_create = False
                raise SchedulerApiError("timeout", status_code=None)
            return copy.deepcopy(job)
        if method == "GET":
            if tail not in self.jobs:
                raise SchedulerApiError("not found", status_code=404)
            return copy.deepcopy(self.jobs[tail])
        if method == "DELETE":
            if tail not in self.jobs:
                raise SchedulerApiError("not found", status_code=404)
            del self.jobs[tail]
            return {}
        raise AssertionError(f"unexpected {method} {url}")

    def posts(self) -> list[tuple[str, str, dict[str, Any] | None]]:
        return [c for c in self.calls if c[0] == "POST"]


@pytest.fixture
def batch_server() -> FakeBatchServer:
    return FakeBatchServer()


@pytest.fixture
def k8s_server() -> FakeK8sServer:
    return FakeK8sServer()


@pytest.fixture
def no_sleep() -> list[float]:
    return []
