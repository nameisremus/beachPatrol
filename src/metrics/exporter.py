from __future__ import annotations

import inspect
import logging
import os
import time
import functools
import aiohttp
from functools import wraps
import requests
from typing import Any, Callable
from urllib.parse import urlparse
from celery import signals
import redis

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Info,
    generate_latest,
)

from config import (
    ENV,
    GIT_BRANCH,
    GIT_COMMIT,
    GIT_TAG,
    METRICS_PORT,
    PROMETHEUS_PREFIX,
    VERSION,
    sensitive_vars,
)
from .server import MetricsServer

LOG = logging.getLogger(__name__)


class MetricRegistry:
    registry = CollectorRegistry()

    # Static info
    build_info = Info(
        f"{PROMETHEUS_PREFIX}build_info",
        "Build metadata",
        registry=registry,
    )
    env_info = Gauge(
        f"{PROMETHEUS_PREFIX}env_info",
        "Environment variables (non-secret)",
        ["key", "value"],
        registry=registry,
    )

    # Bot commands
    http_total = Counter(
       f"{PROMETHEUS_PREFIX}http_requests_total",
        "Slash-command requests grouped by interface",
        ["handler", "interface", "code"],
        registry=registry,
    )
    http_latency = Histogram(
        f"{PROMETHEUS_PREFIX}http_request_latency_seconds",
        "Latency of bot commands",
        ["handler"],
        registry=registry,
    )

    # UI interactions
    ui_total = Counter(
        f"{PROMETHEUS_PREFIX}interaction_total",
        "UI interactions (buttons etc.)",
        ["component", "interface"],
        registry=registry,
    )
    ui_latency = Histogram(
        f"{PROMETHEUS_PREFIX}interaction_latency_seconds",
        "Latency of UI interactions",
        ["component"],
        registry=registry,
    )

    # Celery
    celery_total = Counter(
        f"{PROMETHEUS_PREFIX}celery_task_total",
        "Celery task executions",
        ["task", "state"],
        registry=registry,
    )
    celery_runtime = Histogram(
        f"{PROMETHEUS_PREFIX}celery_task_runtime_seconds",
        "Celery task runtime",
        ["task"],
        registry=registry,
    )

    # Outgoing HTTP
    ext_total = Counter(
        f"{PROMETHEUS_PREFIX}external_http_total",
        "Outgoing HTTP requests",
        ["service", "method", "code"],
        registry=registry,
    )

    # Redis
    redis_ops = Counter(
        f"{PROMETHEUS_PREFIX}redis_ops_total",
        "Counted Redis operations",
        ["kind"],
        registry=registry,
    )

def _determine_interface(name: str) -> str:
        # Return 'discord' if name starts with dc_, 'telegram' if tg_, else unknown
        if name.startswith("dc_"):
            return "discord"
        if name.startswith("tg_"):
            return "telegram"
        return "unknown"

class _Exporter:
    def __init__(self) -> None:
        self.m = MetricRegistry
        self._populate_static_info()
        self._patch_requests_if_enabled()
        self._patch_redis_if_enabled()
        self._patch_http_clients()

        MetricsServer(METRICS_PORT, self.render_latest).start()

    def _populate_static_info(self) -> None:
        # build_info and env_info
        self.m.build_info.info({
            "environment": ENV,
            "branch": GIT_BRANCH,
            "tag": GIT_TAG,
            "commit": GIT_COMMIT,
            "version": VERSION,
        })
        for key, val in os.environ.items():
            if key not in sensitive_vars:
                self.m.env_info.labels(key=key, value=val).set(1)

    @staticmethod
    def track_http(handler_name: str | None = None):
        """
        Decorator for slash/HTTP-style commands.
        """
        def _wrap_callable(fn, name: str):
            interface = _determine_interface(name)

            if inspect.iscoroutinefunction(fn):

                @functools.wraps(fn)
                async def _async(*a, **kw):
                    try:
                        return await fn(*a, **kw)
                    finally:
                        MetricRegistry.http_total.labels(
                            handler=name, interface=interface, code="200"
                        ).inc()
                return _async

            @functools.wraps(fn)
            def _sync(*a, **kw):
                try:
                    return fn(*a, **kw)
                finally:
                    MetricRegistry.http_total.labels(
                        handler=name, interface=interface, code="200"
                    ).inc()
            return _sync

        def decorator(obj):
            # Discord slash-command object
            try:
                from discord import app_commands
                is_discord_cmd = isinstance(obj, app_commands.Command)
            except Exception:
                is_discord_cmd = False

            if is_discord_cmd:
                name = handler_name or obj.name or obj.callback.__name__
                wrapped = _wrap_callable(obj.callback, name)
                #   .callback is a read-only property; overwrite the
                #   underlying storage attribute used by the library.
                obj._callback = wrapped
                return obj

            # Plain coroutine / function
            name = handler_name or obj.__name__
            return _wrap_callable(obj, name)

        return decorator


    def track_interaction(self, component: str):
        # Decorator for button / select / pagination callbacks (auto-detect)

        interface = _determine_interface(component)

        def decorator(fn):
            if inspect.iscoroutinefunction(fn):

                @wraps(fn)
                async def async_wrap(*a, **kw):
                    try:
                        return await fn(*a, **kw)
                    finally:
                        self.m.ui_total.labels(
                            component=component, interface=interface
                        ).inc()

                return async_wrap

            @wraps(fn)
            def sync_wrap(*a, **kw):
                try:
                    return fn(*a, **kw)
                finally:
                    self.m.ui_total.labels(
                        component=component, interface=interface
                    ).inc()

            return sync_wrap

        return decorator


    def track_redis(self, kind: str):
        # Decorator for functions that wrap a Redis operation
        def decorator(fn):
            @wraps(fn)
            def wrap(*a, **kw):
                self.m.redis_ops.labels(kind=kind).inc()
                return fn(*a, **kw)
            return wrap
        return decorator

    def instrument_celery(self, celery_app) -> None:
        # avoid double‐wiring
        if getattr(celery_app, "_bp_instrumented", False):
            return
        celery_app._bp_instrumented = True

        # 1) hook your prerun/postrun in this (master or child) process
        @signals.task_prerun.connect
        def _on_start(sender=None, task=None, **_kw):
            task.__bp_start__ = time.time()

        @signals.task_postrun.connect
        def _on_done(sender=None, task=None, state=None, **_kw):
            elapsed = time.time() - getattr(task, "__bp_start__", time.time())
            self.m.celery_total.labels(task=sender, state=state or "UNKNOWN").inc()
            self.m.celery_runtime.labels(task=sender).observe(elapsed)

        # 2) when Celery forks, re-attach in each child
        @signals.worker_process_init.connect
        def _on_worker_init(**_kw):
            # re-run the same wiring, but guarded by _bp_instrumented
            self.instrument_celery(celery_app)

    def _patch_requests_if_enabled(self) -> None:
        # Monkey-patch requests.Session.request unless we're in development
        if ENV.lower() == "development":
            LOG.info("Skipping requests patch in development environment")
            return

        if getattr(requests, "_bp_patched", False):
            return

        original_request = requests.sessions.Session.request
        metrics = self.m

        @functools.wraps(original_request)
        def patched(session_self, method, url, *a, **kw):
            service = urlparse(url).hostname or "unknown"
            start   = time.time()
            http_code = "error"

            try:
                response  = original_request(session_self, method, url, *a, **kw)
                http_code = str(response.status_code)
                return response
            except requests.Timeout:
                http_code = "timeout"
                raise
            finally:
                method_upper = method.upper()
                metrics.ext_total.labels(
                    service=service, method=method_upper, code=http_code
                ).inc()

        requests.sessions.Session.request = patched
        requests._bp_patched = True

    def _patch_http_clients(self) -> None:
        """Monkey-patch aiohttp, httpx, and openai so that every async HTTP call is counted."""
        metrics = self.m

        # ─── aiohttp ─────────────────────────────────────────────────────────
        try:
            original_aio = aiohttp.ClientSession._request

            @functools.wraps(original_aio)
            async def _patched_aio(self, method, url, *args, **kwargs):
                service = urlparse(str(url)).hostname or "unknown"
                http_code = "error"
                try:
                    resp = await original_aio(self, method, url, *args, **kwargs)
                    http_code = str(getattr(resp, "status", "0"))
                    return resp
                finally:
                    metrics.ext_total.labels(
                        service=service,
                        method=method.upper(),
                        code=http_code,
                    ).inc()

            aiohttp.ClientSession._request = _patched_aio
        except ImportError:
            LOG.debug("aiohttp not installed; skipping aiohttp patch")

        # ─── httpx ───────────────────────────────────────────────────────────
        try:
            import httpx

            original_httpx = httpx.AsyncClient.request

            @functools.wraps(original_httpx)
            async def _patched_httpx(self, method, url, *args, **kwargs):
                service = urlparse(str(url)).hostname or "unknown"
                http_code = "error"
                try:
                    resp = await original_httpx(self, method, url, *args, **kwargs)
                    http_code = str(getattr(resp, "status_code", "0"))
                    return resp
                finally:
                    metrics.ext_total.labels(
                        service=service,
                        method=method.upper(),
                        code=http_code,
                    ).inc()

            httpx.AsyncClient.request = _patched_httpx
        except ImportError:
            LOG.debug("httpx not installed; skipping httpx patch")

        try:
            import openai
            from openai import api_requestor

            orig = api_requestor.APIRequestor.request

            @functools.wraps(orig)
            def _patched_openai(self, method, url, *args, **kwargs):
                service = urlparse(str(url)).hostname or "unknown"
                http_code = "error"
                try:
                    result = orig(self, method, url, *args, **kwargs)
                    # result may be (resp, data, api_key) or similar
                    resp = result[0] if isinstance(result, (tuple, list)) else result
                    http_code = str(getattr(resp, "status_code", getattr(resp, "status", "0")))
                    return result
                finally:
                    metrics.ext_total.labels(
                        service=service,
                        method=method.upper(),
                        code=http_code,
                    ).inc()

            api_requestor.APIRequestor.request = _patched_openai
        except ImportError:
            LOG.debug("openai not installed; skipping openai patch")


    def _patch_redis_if_enabled(self) -> None:
        try:
            if getattr(redis, "_bp_redis_patched", False):
                return

            # wrap the core methods you use
            for method in ("get", "set", "scan_iter", "rpush", "lrange"):
                original = getattr(redis.Redis, method, None)
                if not original:
                    continue

                wrapped = self.track_redis(method)(original)
                setattr(redis.Redis, method, wrapped)

            # guard so we don’t double-wrap
            redis._bp_redis_patched = True
        except ImportError:
            LOG.debug("redis not installed; skipping redis patch")
            return

    def render_latest(self) -> bytes:
        # Return latest metrics in Prometheus text format
        return generate_latest(self.m.registry)


# Instantiate the exporter and start the metrics HTTP server
_exporter = _Exporter()

# tracking
track_http = _exporter.track_http
track_interaction = _exporter.track_interaction
track_redis = _exporter.track_redis
instrument_celery = _exporter.instrument_celery
