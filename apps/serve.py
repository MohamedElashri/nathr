"""Terminal and loopback HTTP frontends for the Nathr JSONL worker."""

import argparse
import base64
from collections import deque
from contextlib import ExitStack, closing, contextmanager
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import select
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import parse_qs, unquote, urlsplit
from uuid import uuid4


class RequestError(Exception):
    def __init__(self, message, status=400):
        super().__init__(message)
        self.status = status


def json_bytes(value):
    return json.dumps(value, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def read_json(raw):
    def object_pairs(pairs):
        result = {}
        for key, value in pairs:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result

    def invalid_constant(value):
        raise ValueError(f"invalid JSON value: {value}")

    return json.loads(raw.decode("utf-8"), object_pairs_hook=object_pairs, parse_constant=invalid_constant)


def check_fields(value, allowed):
    if extra := value.keys() - allowed:
        raise RequestError(f"unsupported option: {min(extra)}")


def token_input(value):
    return isinstance(value, list) and bool(value) and all(type(item) is int for item in value)


def prompt_input(value):
    if isinstance(value, str) and value:
        return {"prompt": value}
    if token_input(value):
        return {"tokens": value}
    raise RequestError("expected a nonempty string or token array")


def completion_usage(event):
    prompt, generated = event["prompt_tokens"], event["emitted"]
    usage = {"prompt_tokens": prompt, "completion_tokens": generated, "total_tokens": prompt + generated}
    if cached := event.get("cached_prompt_tokens"):
        usage["prompt_tokens_details"] = {"cached_tokens": cached}
    return usage


def chat_logprob(score):
    def display(item):
        return {key: item[key] for key in ("token", "bytes", "logprob")}

    return dict(display(score), top_logprobs=[display(item) for item in score["top_logprobs"]])


def response_message(value):
    if not isinstance(value, dict) or value.get("type", "message") != "message":
        raise RequestError("Responses input items must be text messages")
    role, content = value.get("role"), value.get("content")
    if role not in ("system", "developer", "user", "assistant"):
        raise RequestError("unsupported message role")
    if isinstance(content, list) and content:
        if not all(isinstance(part, dict) and part.get("type") in ("input_text", "output_text") and
                   isinstance(part.get("text"), str) for part in content):
            raise RequestError("Responses content must contain text parts")
        content = "".join(part["text"] for part in content)
    if not isinstance(content, str):
        raise RequestError("message content must be text")
    return {"role": "system" if role == "developer" else role, "content": content}


def function_name(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", value):
        raise RequestError("invalid function name")
    return value


def structured_format(value, responses=False):
    if value is None:
        return False, None
    if not isinstance(value, dict):
        raise RequestError("output format must be an object")
    kind = value.get("type")
    if kind in ("text", "json_object"):
        check_fields(value, {"type"})
        return kind == "json_object", None
    if kind != "json_schema":
        raise RequestError("unsupported output format")
    spec = value if responses else value.get("json_schema")
    if not responses:
        check_fields(value, {"type", "json_schema"})
    if not isinstance(spec, dict) or not isinstance(spec.get("schema"), dict):
        raise RequestError("json_schema requires a schema object")
    check_fields(spec, {"name", "schema", "description", "strict"} | ({"type"} if responses else set()))
    function_name(spec.get("name"))
    return True, spec["schema"]


def tool_history(messages):
    if not isinstance(messages, list):
        return messages
    mapped, pending = [], None
    for message in messages:
        if not isinstance(message, dict):
            raise RequestError("chat messages must be objects")
        role, content = message.get("role"), message.get("content")
        if pending is not None:
            call_id, name = pending
            if role != "tool" or message.get("tool_call_id") != call_id or not isinstance(content, str):
                raise RequestError("expected a text tool result with the matching tool_call_id")
            mapped.append({"role": "user", "content": f"Function result for {name}:\n{content}"})
            pending = None
        elif role == "tool":
            raise RequestError("tool result has no matching assistant tool call")
        elif role == "assistant" and message.get("tool_calls") not in (None, []):
            calls = message["tool_calls"]
            if not isinstance(calls, list) or len(calls) != 1 or not isinstance(calls[0], dict):
                raise RequestError("one assistant function call per turn is supported")
            call = calls[0]
            call_id, function = call.get("id"), call.get("function")
            if (call.get("type") != "function" or not isinstance(function, dict) or
                    not isinstance(call_id, str) or not 1 <= len(call_id) <= 128 or content not in (None, "")):
                raise RequestError("invalid assistant function call")
            name, arguments = function_name(function.get("name")), function.get("arguments")
            if not isinstance(arguments, str):
                raise RequestError("function arguments must be a JSON object string")
            try:
                arguments = read_json(arguments.encode())
            except (ValueError, RecursionError) as error:
                raise RequestError("invalid JSON function arguments") from error
            if not isinstance(arguments, dict):
                raise RequestError("function arguments must be a JSON object string")
            call_text = json_bytes({"name": name, "arguments": arguments}).decode()
            mapped.append({"role": "assistant", "content": f"Function call:\n{call_text}"})
            pending = (call_id, name)
        else:
            mapped.append(message)
    if pending is not None:
        raise RequestError("assistant function call requires a matching tool result")
    return mapped


def chat_tool(value):
    tools = value.pop("tools", None)
    choice = value.pop("tool_choice", None)
    parallel = value.pop("parallel_tool_calls", None)
    if parallel is not None and parallel is not False:
        raise RequestError("parallel tool calls are unsupported")
    if tools is None and choice is None:
        return None
    if choice == "none":
        return None
    if not isinstance(tools, list) or not 1 <= len(tools) <= 16:
        raise RequestError("tools must contain between one and 16 function definitions")
    functions = {}
    for tool in tools:
        if not isinstance(tool, dict) or tool.get("type") != "function" or not isinstance(tool.get("function"), dict):
            raise RequestError("tools must be function definitions")
        check_fields(tool, {"type", "function"})
        function = tool["function"]
        check_fields(function, {"name", "description", "parameters", "strict"})
        name = function_name(function.get("name"))
        if name in functions:
            raise RequestError("duplicate function name")
        functions[name] = function
    if choice == "required" and len(functions) == 1:
        selected = next(iter(functions))
    elif isinstance(choice, dict) and choice.get("type") == "function" and isinstance(choice.get("function"), dict):
        check_fields(choice, {"type", "function"})
        check_fields(choice["function"], {"name"})
        selected = function_name(choice["function"].get("name"))
    else:
        raise RequestError("select a named function using tool_choice")
    if selected not in functions:
        raise RequestError("tool_choice must select a declared function")
    return functions[selected]


def model_card(model_id, parent=None):
    card = {"id": model_id, "object": "model", "owned_by": "nathr", "created": 0}
    if parent:
        card.update(parent=parent, root=parent)
    return card


def checked_event(event):
    if event["event"] == "error" or event.get("reason") == "error":
        status = 429 if event.get("code") == "resource_limit" else 400
        raise RequestError(event.get("message", "inference failed"), status)
    return event


@dataclass
class Request:
    id: int
    payload: bytes | None
    events: deque = field(default_factory=deque)
    buffered: int = 0
    sent: bool = False
    cancelled: bool = False
    cancel_sent: bool = False
    error: RequestError | None = None
    submitted_at: float = field(default_factory=time.monotonic)


class Worker:
    def __init__(self, command, limits, static_adapters=None, metrics=None):
        self.limits = limits
        self.metrics = metrics
        self.changed = threading.Condition()
        self.pending = {}
        self.next_id = 0
        self.cancelled = 0
        self.queue_bypasses = 0
        self.kv = {"peak_bytes": 0, "resident_bytes": 0}
        self.metadata = None
        self.static_adapters = static_adapters or {}
        self.lora_adapters = {}
        self.outputs = set()
        self.error = None
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        # Buffered reads keep the pipe efficient and cap each worker message.
        self.stdout = open(self.process.stdout.fileno(), "rb", closefd=False)
        self.threads = [threading.Thread(target=work, daemon=True) for work in (self.read, self.write)]
        for thread in self.threads:
            thread.start()

    def start(self, check=None):
        deadline = time.monotonic() + 60
        with self.changed:
            while self.metadata is None and self.error is None:
                if check is not None:
                    check()
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    raise RuntimeError("model did not become ready")
                self.changed.wait(min(remaining, 0.1))
            if self.error:
                raise self.error

    def idle(self):
        with self.changed:
            return not self.pending

    def fail(self, message):
        with self.changed:
            if self.error:
                return
            self.error = RequestError(message, 503)
            for request in self.pending.values():
                request.error = self.error
            self.pending.clear()
            self.changed.notify_all()

    def close(self):
        self.fail("engine is shutting down")
        if self.process.poll() is None:
            self.process.terminate()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.kill()
                self.process.wait()
        for thread in self.threads:
            thread.join()
        self.stdout.close()
        self.process.stdin.close()
        self.process.stdout.close()

    def submit(self, value):
        with self.changed:
            if self.error:
                raise self.error
            if len(self.pending) >= self.limits.max_clients:
                raise RequestError("request limit reached", 429)
            self.next_id += 1
            payload = json_bytes(dict(value, id=self.next_id))
            if len(payload) > 1024 * 1024:
                raise RequestError("request exceeds the worker's input limit", 413)
            request = Request(self.next_id, payload + b"\n")
            self.pending[request.id] = request
            self.changed.notify_all()
            return request

    def cancel_locked(self, request):
        if request.id not in self.pending or request.cancelled:
            return
        request.cancelled = True
        request.events.clear()
        request.buffered = 0
        if not request.sent:
            del self.pending[request.id]
            self.cancelled += 1
        self.changed.notify_all()

    def cancel(self, request):
        with self.changed:
            self.cancel_locked(request)

    def take(self, request, timeout):
        with self.changed:
            self.changed.wait_for(lambda: request.events or request.error, timeout=timeout)
            if request.error:
                raise request.error
            if request.events:
                value, size = request.events.popleft()
                request.buffered -= size
                return value
            return None

    def read(self):
        try:
            # A scored token frame can include its own bytes and up to 20 alternatives.
            cap = max(64 * self.limits.max_result_values + 4096, 256 * 1024)
            while line := self.stdout.readline(cap + 1):
                if len(line) > cap or not line.endswith(b"\n"):
                    raise RuntimeError("worker response exceeds its frame limit")
                event = json.loads(line)
                with self.changed:
                    if event["event"] == "ready":
                        del event["event"]
                        self.lora_adapters = {name: (self.static_adapters[name], False)
                                              for name in event.pop("lora_adapters", ())}
                        self.metadata = event
                        self.outputs = {item["name"] for item in event["outputs"]}
                        self.changed.notify_all()
                        continue
                    size = len(line)
                    if event["event"] == "result" and "lora_adapters" in event:
                        names = event.pop("lora_adapters")
                        loaded = event.pop("lora_loaded", None)
                        adapters = dict(self.lora_adapters)
                        if loaded is not None:
                            adapters[loaded["name"]] = (loaded["path"], True)
                        self.lora_adapters = {name: adapters[name] for name in names}
                        size = len(json_bytes(event))
                    if event["event"] == "metrics":
                        if self.metrics is not None:
                            self.metrics.scheduler = event["values"]
                        continue
                    request = self.pending.get(event.get("id"))
                    if request is None:
                        continue
                    terminal = event["event"] in {"done", "error", "result"}
                    if self.metrics is not None:
                        duration = time.monotonic() - request.submitted_at if terminal else None
                        self.metrics.observe(event, duration)
                    if terminal:
                        if "kv" in event:
                            self.kv = event["kv"]
                            self.queue_bypasses = event["queue_bypasses"]
                        del self.pending[request.id]
                        self.cancelled += event.get("reason") == "cancelled"
                    if request.cancelled:
                        continue
                    if request.buffered + size > self.limits.max_buffer_bytes or len(request.events) == 256:
                        request.error = RequestError("response buffer limit reached", 429)
                        self.cancel_locked(request)
                    else:
                        request.events.append((event, size))
                        request.buffered += size
                    self.changed.notify_all()
            self.fail("model worker exited")
        except (OSError, ValueError, KeyError, RuntimeError) as error:
            self.fail(f"cannot read model worker: {error}")

    def write(self):
        try:
            while True:
                with self.changed:
                    payload = None
                    while payload is None and not self.error:
                        request = next(
                            (item for item in self.pending.values() if item.cancelled and not item.cancel_sent),
                            None,
                        )
                        if request:
                            request.cancel_sent = True
                            payload = json_bytes({"cancel": request.id}) + b"\n"
                            continue
                        request = next((item for item in self.pending.values() if not item.sent), None)
                        if request:
                            request.sent = True
                            payload, request.payload = request.payload, None
                        else:
                            self.changed.wait()
                    if self.error:
                        return
                data = memoryview(payload)
                while data:
                    written = self.process.stdin.write(data)
                    if not written:
                        raise OSError("model worker input is closed")
                    data = data[written:]
        except (OSError, ValueError) as error:
            self.fail(f"cannot write model worker: {error}")

    def health(self):
        with self.changed:
            return dict(
                status="unavailable" if self.error else "ok",
                pending_requests=len(self.pending),
                cancelled_requests=self.cancelled,
                queue_bypasses=self.queue_bypasses,
                kv=self.kv,
            )

    def metrics_snapshot(self):
        with self.changed:
            return self.metrics.snapshot(up=self.error is None, pending=len(self.pending), cancelled=self.cancelled)


@dataclass
class WorkerSlot:
    path: Path
    source: str | None
    worker: Worker | None = None
    plan: dict | None = None
    leases: int = 0
    last_used: float = 0
    adapters: dict = field(default_factory=dict)


def checkpoint_command(command, check):
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                          text=True, start_new_session=True) as process:
        try:
            while True:
                check()
                try:
                    result, stderr = process.communicate(timeout=0.1)
                    break
                except subprocess.TimeoutExpired:
                    pass
        except BaseException:
            try:
                os.killpg(process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                process.communicate(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                process.communicate()
            raise
        if process.returncode:
            raise RuntimeError(stderr.strip() or "checkpoint command failed")
        return result


class WorkerPool:
    def __init__(self, executable, store, slots, options, worker_args, metrics_factory):
        self.executable = executable
        self.store = store
        self.slots = slots
        self.options = options
        self.worker_args = worker_args
        self.metrics_factory = metrics_factory
        self.budget = (getattr(options, "memory_budget_mib", None) or 0) * 1024 * 1024
        self.changed = threading.Condition()
        self.stopping = threading.Event()
        self.loading = None

    def primary_adapters(self):
        primary = self.slots[self.options.model_id]
        return primary.worker.lora_adapters if primary.worker is not None else primary.adapters

    def resolve(self, model_id):
        if isinstance(model_id, str):
            with self.changed:
                if model_id in self.slots:
                    return model_id, self.slots[model_id]
                if model_id in self.primary_adapters():
                    primary = self.options.model_id
                    return primary, self.slots[primary]
        raise RequestError("model is not configured", 404)

    def model_cards(self):
        with self.changed:
            return [model_card(name) for name in self.slots] + [
                model_card(name, self.options.model_id) for name in sorted(self.primary_adapters())]

    def reserved_bytes(self):
        return sum(slot.plan["total_bytes"] for slot in self.slots.values() if slot.plan is not None)

    def reserve(self, slot, plan, check):
        size = plan["total_bytes"]
        if size > self.budget:
            raise RequestError(f"checkpoint needs {size} bytes, above the server memory budget of {self.budget} bytes",
                               429)
        victims = []
        with self.changed:
            need = self.reserved_bytes() + size - self.budget
            if need > 0:
                idle = sorted((item for item in self.slots.values()
                               if item.worker is not None and not item.leases and item.worker.idle()),
                              key=lambda item: item.last_used)
                for item in idle:
                    victims.append((item, item.worker))
                    need -= item.plan["total_bytes"]
                    if need <= 0:
                        break
                if need > 0:
                    raise RequestError("active requests are occupying the server memory budget", 429)
                for item, worker in victims:
                    item.adapters = worker.lora_adapters
                    item.worker = None
        for item, worker in victims:
            worker.close()
            with self.changed:
                item.plan = None
        check()
        with self.changed:
            slot.plan = plan

    def load(self, model_id, slot, check):
        worker = None
        try:
            if slot.source is not None:
                if self.budget:
                    checkpoint_command([self.executable, "pull", slot.source, "--models-dir",
                                        str(self.store.directory)], check)
                else:
                    self.store.pull(slot.source)
            command = [self.executable, "batch", "--model", str(slot.path), *self.worker_args]
            static_adapters = {name: path for name, (path, runtime) in slot.adapters.items() if not runtime}
            if model_id == self.options.model_id:
                if self.options.tokenizer:
                    command.extend(("--tokenizer", self.options.tokenizer))
                if self.options.chat_format != "auto":
                    command.extend(("--chat-format", self.options.chat_format))
                for name, path in static_adapters.items():
                    command.extend(("--lora", f"{name}={path}"))
                if self.options.lora_root is not None:
                    command.extend(("--lora-root", self.options.lora_root))
            if self.budget:
                plan = json.loads(checkpoint_command([*command, "--memory-plan"], check))
                self.reserve(slot, plan, check)
            metrics = self.metrics_factory() if self.metrics_factory else None
            worker = Worker([*command, "--ready"], self.options, static_adapters, metrics)
            worker.start(check)
            for name, (path, runtime) in slot.adapters.items():
                if runtime:
                    self.restore_adapter(worker, name, path, check)
            check()
            if self.budget and (worker.metadata["weight_bytes"] != plan["weights_bytes"] or
                                worker.metadata["vocabulary"] > plan["vocabulary_limit"]):
                raise RequestError("checkpoint size or vocabulary changed during loading", 503)
            return worker
        except BaseException as error:
            if worker is not None:
                worker.close()
            with self.changed:
                slot.plan = None
            if isinstance(error, (OSError, RuntimeError, ValueError)) and not isinstance(error, ConnectionError):
                raise RequestError(f"{model_id}: {error}", 503) from error
            raise

    @staticmethod
    def restore_adapter(worker, name, path, check):
        request = worker.submit({"load_lora": {"name": name, "path": path}})
        try:
            while True:
                check()
                event = worker.take(request, 0.1)
                if event is not None:
                    checked_event(event)
                    return
        finally:
            worker.cancel(request)

    @contextmanager
    def borrow(self, model_id, deadline=None, check_peer=None):
        model_id, slot = self.resolve(model_id)

        def check():
            if self.stopping.is_set():
                raise RequestError("engine is shutting down", 503)
            if deadline is not None and time.monotonic() >= deadline:
                raise RequestError("model load deadline exceeded", 504)
            if check_peer is not None:
                check_peer()

        loading = False
        try:
            with self.changed:
                while True:
                    check()
                    if slot.worker is not None:
                        worker = slot.worker
                        slot.leases += 1
                        break
                    if self.loading is None:
                        loading = True
                        self.loading = model_id
                        break
                    self.changed.wait(0.1)
            if loading:
                worker = self.load(model_id, slot, check)
                with self.changed:
                    slot.worker = worker
                    slot.leases += 1
        finally:
            if loading:
                with self.changed:
                    self.loading = None
                    self.changed.notify_all()
        try:
            yield worker
        finally:
            with self.changed:
                slot.leases -= 1
                slot.last_used = time.monotonic()

    def health(self, model_id=None):
        def snapshot(name, slot):
            if slot.worker is not None:
                value = slot.worker.health()
            else:
                state = "loading" if self.loading == name else "evicting" if slot.plan else "unloaded"
                value = {"status": state, "pending_requests": 0, "cancelled_requests": 0,
                         "queue_bypasses": 0, "kv": {"peak_bytes": 0, "resident_bytes": 0}}
            if self.budget:
                value["model_memory"] = slot.plan
            return value

        with self.changed:
            if model_id is not None:
                name, slot = self.resolve(model_id)
                result = snapshot(name, slot)
            elif len(self.slots) == 1:
                name, slot = next(iter(self.slots.items()))
                result = snapshot(name, slot)
            else:
                models = {name: snapshot(name, slot) for name, slot in self.slots.items()}
                failed = sum(value["status"] == "unavailable" for value in models.values())
                status = "unavailable" if failed == len(models) else "degraded" if failed else "ok"
                result = {"status": status, "models": models}
            if self.budget:
                result["memory_budget"] = {"budget_bytes": self.budget, "reserved_bytes": self.reserved_bytes()}
            return result

    def metrics(self, model_id):
        with self.changed:
            if model_id is None:
                slots = self.slots.items()
            else:
                slots = (self.resolve(model_id),)
            return {
                name: slot.worker.metrics_snapshot() if slot.worker is not None
                else self.metrics_factory().snapshot(up=False, pending=0, cancelled=0)
                for name, slot in slots
            }

    def close(self):
        self.stopping.set()
        with self.changed:
            self.changed.notify_all()
            self.changed.wait_for(lambda: self.loading is None)
        for slot in self.slots.values():
            if slot.worker is not None:
                slot.worker.close()


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, pool, options):
        directory = Path(__file__).with_name("nathr-web")
        self.assets = {route: ((directory / name).read_bytes(), content_type) for route, name, content_type in (
            ("/", "index.html", "text/html; charset=utf-8"),
            ("/app.css", "app.css", "text/css; charset=utf-8"),
            ("/app.js", "app.js", "text/javascript; charset=utf-8"),
        )}
        super().__init__(("127.0.0.1", options.port), Handler)
        self.pool = pool
        self.default_model = options.model_id
        self.options = options
        self.connections = threading.BoundedSemaphore(options.max_clients + 1)
        self.inference_slots = threading.BoundedSemaphore(options.max_clients)

    def metrics_body(self, model_id):
        from serving_metrics import render_metrics

        return render_metrics(self.pool.metrics(model_id))

    def process_request(self, request, client_address):
        if self.connections.acquire(blocking=False):
            try:
                super().process_request(request, client_address)
            except Exception:
                self.connections.release()
                self.shutdown_request(request)
                raise
        else:
            try:
                request.settimeout(0.1)
                request.sendall(b"HTTP/1.1 429 Too Many Requests\r\nConnection: close\r\nContent-Length: 0\r\n\r\n")
            except OSError:
                pass
            self.shutdown_request(request)

    def process_request_thread(self, request, client_address):
        try:
            super().process_request_thread(request, client_address)
        finally:
            self.connections.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.options.io_timeout)
        self.started_stream = False
        self.response_sequence = None
        self.next_peer_check = 0

    def log_message(self, *args):
        pass

    def check_host(self):
        port = self.server.server_address[1]
        hosts = {name for base in ("127.0.0.1", "localhost") for name in (base, f"{base}:{port}")}
        values = self.headers.get_all("Host", [])
        if len(values) != 1 or values[0].lower() not in hosts:
            raise RequestError("expected a loopback Host header")

    def reply(self, value, status=200):
        self.send_body(json_bytes(value), "application/json", status)

    def check_response_size(self, size):
        if size > self.server.options.max_buffer_bytes:
            raise RequestError("response buffer limit reached", 429)

    def inference_body(self, body):
        self.check_response_size(len(body))
        self.send_body(body, "application/json")

    def inference_reply(self, value):
        self.inference_body(json_bytes(value))

    def send_body(self, body, content_type, status=200, ui=False):
        self.close_connection = True
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Connection", "close")
        if ui:
            self.send_header("Cache-Control", "no-cache")
            self.send_header("X-Content-Type-Options", "nosniff")
            self.send_header("Content-Security-Policy", "default-src 'none'; script-src 'self'; style-src 'self'; "
                             "connect-src 'self'; base-uri 'none'; form-action 'self'; frame-ancestors 'none'")
        self.end_headers()
        self.wfile.write(body)

    def send_error_response(self, error):
        response = {"error": {"message": str(error), "type": "request_error", "code": error.status}}
        try:
            if self.response_sequence is not None:
                self.response_event("error", message=str(error), code=str(error.status), param=None, **response)
            elif self.started_stream:
                self.sse(response, enforce_limit=False)
                self.sse("[DONE]")
            else:
                self.reply(response, error.status)
        except OSError:
            self.close_connection = True

    def do_GET(self):
        try:
            self.check_host()
            target = urlsplit(self.path)
            path = target.path
            query = parse_qs(target.query, keep_blank_values=True)
            models = query.get("model", [])
            if len(models) > 1:
                raise RequestError("select one model")
            model_id = models[0] if models else None
            if path in self.server.assets:
                self.send_body(*self.server.assets[path], ui=True)
            elif path == "/health":
                health = self.server.pool.health(model_id)
                self.reply(health, 503 if health["status"] == "unavailable" else 200)
            elif path == "/metrics":
                self.send_body(self.server.metrics_body(model_id), "text/plain; version=0.0.4; charset=utf-8")
            elif path == "/v1/models":
                cards = self.server.pool.model_cards()
                self.reply({"object": "list", "data": cards})
            elif path.startswith("/v1/models/"):
                name = unquote(path.removeprefix("/v1/models/"))
                parent, _ = self.server.pool.resolve(name)
                self.reply(model_card(name, parent if parent != name else None))
            elif path == "/model":
                name = self.server.default_model if model_id is None else model_id
                options = self.server.options
                limits = {name: getattr(options, name) for name in
                          ("max_body_bytes", "max_output_tokens", "max_result_values", "max_buffer_bytes")}
                parent, _ = self.server.pool.resolve(name)
                deadline = time.monotonic() + options.request_timeout
                with self.server.pool.borrow(parent, deadline, self.check_peer) as worker:
                    metadata = dict(worker.metadata, id=name, limits=limits)
                    if name != parent:
                        metadata["base_model"] = parent
                    self.reply(metadata)
            else:
                raise RequestError("endpoint not found", 404)
        except RequestError as error:
            self.send_error_response(error)
        except ValueError:
            self.send_error_response(RequestError("invalid request path"))
        except OSError:
            self.close_connection = True

    def body(self):
        if self.headers.get("Transfer-Encoding") is not None:
            raise RequestError("send a Content-Length body")
        lengths = self.headers.get_all("Content-Length", [])
        if not lengths:
            raise RequestError("Content-Length is required", 411)
        if len(lengths) != 1 or not lengths[0].isascii() or not lengths[0].isdecimal():
            raise RequestError("invalid Content-Length")
        if len(lengths[0]) > 12 or int(lengths[0]) > self.server.options.max_body_bytes:
            raise RequestError("request body limit exceeded", 413)
        remaining = int(lengths[0])
        chunks = bytearray()
        deadline = time.monotonic() + self.server.options.io_timeout
        while remaining:
            budget = deadline - time.monotonic()
            if budget <= 0:
                raise RequestError("request body deadline exceeded", 408)
            self.connection.settimeout(budget)
            chunk = self.rfile.read1(min(8192, remaining))
            if not chunk:
                raise RequestError("incomplete request body")
            chunks.extend(chunk)
            remaining -= len(chunk)
        self.connection.settimeout(self.server.options.io_timeout)
        try:
            value = read_json(chunks)
        except (ValueError, UnicodeError, RecursionError) as error:
            raise RequestError(f"invalid JSON: {error}") from error
        if not isinstance(value, dict):
            raise RequestError("request body must be a JSON object")
        return value

    def normalize(self, value, kind):
        options, metadata = self.server.options, self.worker.metadata
        stream = value.pop("stream", False)
        if kind != "infer" and stream is None:
            stream = False
        if not isinstance(stream, bool):
            raise RequestError("stream must be a boolean")
        include_usage = False
        if kind != "infer":
            allowed = {"max_tokens", "temperature", "top_p", "stop", "seed", "n", "stream_options", "user"}
            chat = {"messages", "max_completion_tokens", "logprobs", "top_logprobs"}
            check_fields(value, allowed | (chat if kind == "chat" else {"prompt"}))
            value.pop("user", None)
            count = value.pop("n", None)
            if count is not None and (type(count) is not int or count != 1):
                raise RequestError("only one completion per request is supported")
            stream_options = value.pop("stream_options", None)
            if stream_options is not None:
                if not stream or not isinstance(stream_options, dict):
                    raise RequestError("stream_options requires a streaming request")
                check_fields(stream_options, {"include_usage"})
                include_usage = stream_options.get("include_usage", False)
                if type(include_usage) is not bool:
                    raise RequestError("include_usage must be a boolean")
            if kind == "chat":
                if "messages" not in value or not metadata["chat"]:
                    raise RequestError("chat requires messages and a checkpoint with a supported chat template")
                logprobs = value.pop("logprobs", None)
                top = value.pop("top_logprobs", None)
                if logprobs is not None and type(logprobs) is not bool:
                    raise RequestError("logprobs must be a boolean")
                if top is not None and (not logprobs or type(top) is not int or not 0 <= top <= 20):
                    raise RequestError("top_logprobs requires logprobs=true and a value from 0 to 20")
                if logprobs:
                    value["top_logprobs"] = top if top is not None else 0
            else:
                if not metadata["text"]:
                    raise RequestError("text completions require a tokenizer")
                value.update(prompt_input(value.pop("prompt", None)))
                if "tokens" in value:
                    value["decode"] = True
            max_tokens = value.pop("max_tokens", None)
            max_completion = value.pop("max_completion_tokens", None)
            if max_tokens is not None and max_completion is not None:
                raise RequestError("choose max_tokens or max_completion_tokens")
            maximum = max_tokens if max_tokens is not None else max_completion
            if maximum is not None:
                value["max_new_tokens"] = maximum
            if (temperature := value.get("temperature")) is not None:
                if type(temperature) not in (float, int) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
                    raise RequestError("temperature must be between 0 and 2")
                value["greedy"] = temperature == 0
                if temperature == 0:
                    value.pop("temperature")
            for key in ("stop", "temperature", "top_p", "seed"):
                if value.get(key) is None:
                    value.pop(key, None)
        if "id" in value or "cancel" in value:
            raise RequestError("request IDs are assigned by the server")
        tensor = "output" in value
        if tensor:
            if stream:
                raise RequestError("named tensors are returned as JSON")
            if not isinstance(value["output"], str) or value["output"] not in self.worker.outputs:
                raise RequestError("checkpoint does not expose the requested output")
        else:
            if not metadata["generation"]:
                raise RequestError("checkpoint requires a named output")
            if "max_new_tokens" in value:
                tokens = value["max_new_tokens"]
                if type(tokens) is not int or not 0 <= tokens <= options.max_output_tokens:
                    raise RequestError("max_new_tokens exceeds the server limit")
        return stream, tensor, include_usage

    def response_input(self, value):
        check_fields(value, {"input", "instructions", "max_output_tokens", "temperature", "top_p", "stream", "store",
                             "text"})
        store = value.get("store")
        if store is not None and store is not False:
            raise RequestError("Responses are stateless. Use store=false")
        text = value.get("input")
        instructions = value.get("instructions")
        if instructions is not None and not isinstance(instructions, str):
            raise RequestError("instructions must be text")
        if isinstance(text, str) and text:
            messages = [{"role": "user", "content": text}]
        elif isinstance(text, list) and text:
            messages = [response_message(item) for item in text]
        else:
            raise RequestError("input must be nonempty text or a list of text messages")
        payload = {key: value[key] for key in ("temperature", "top_p", "stream") if key in value}
        if value.get("max_output_tokens") is not None:
            payload["max_tokens"] = value["max_output_tokens"]
        if self.worker.metadata["chat"]:
            if instructions is not None:
                if messages[0]["role"] == "system":
                    raise RequestError("pass instructions or a system message")
                messages.insert(0, {"role": "system", "content": instructions})
            return dict(payload, messages=messages), "chat"
        if instructions is not None or len(messages) != 1 or messages[0]["role"] != "user":
            raise RequestError("this checkpoint requires a single user prompt")
        return dict(payload, prompt=messages[0]["content"]), "completion"

    def apply_schema(self, value, schema):
        if not self.worker.metadata.get("structured_outputs"):
            raise RequestError("checkpoint does not support structured output")
        if "stop" in value:
            raise RequestError("stop strings cannot be combined with structured output")
        value["json_schema"] = schema

    def check_peer(self):
        if select.select([self.connection], [], [], 0)[0]:
            if not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                raise ConnectionError("client disconnected")

    def waiting(self, request, deadline):
        while (now := time.monotonic()) < deadline:
            if now >= self.next_peer_check:
                self.check_peer()
                self.next_peer_check = now + 0.1
            if result := self.worker.take(request, min(deadline, self.next_peer_check) - now):
                return checked_event(result)
        raise RequestError("inference deadline exceeded", 504)

    def open_stream(self):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.started_stream = True

    def sse(self, value, name=None, *, enforce_limit=True):
        data = b"[DONE]" if value == "[DONE]" else json_bytes(value)
        if enforce_limit:
            self.check_response_size(len(data))
        event = b"" if name is None else b"event: " + name.encode("ascii") + b"\n"
        self.wfile.write(event + b"data: " + data + b"\n\n")
        self.wfile.flush()

    def response_event(self, kind, **fields):
        self.sse(dict(fields, type=kind, sequence_number=self.response_sequence), kind,
                 enforce_limit=kind != "error")
        self.response_sequence += 1

    def generation_events(self, event, request, deadline):
        yield event
        while event["event"] != "done":
            event = self.waiting(request, deadline)
            yield event

    def response_generation(self, first, request, deadline, stream, value):
        identifier = uuid4().hex
        item = {"id": f"msg_{identifier}", "type": "message", "role": "assistant",
                "status": "in_progress", "content": []}
        part = {"type": "output_text", "text": "", "annotations": [], "logprobs": []}
        position = {"item_id": item["id"], "output_index": 0, "content_index": 0}
        response = {"id": f"resp_{identifier}", "object": "response", "created_at": int(time.time()),
                    "model": self.model_id, "status": "in_progress", "output": [], "usage": None,
                    "error": None, "incomplete_details": None, "store": False, "parallel_tool_calls": False,
                    "tool_choice": "none", "tools": [],
                    "text": {"format": (value.get("text") or {}).get("format") or {"type": "text"}},
                    "instructions": value.get("instructions"), "max_output_tokens": value.get("max_output_tokens"),
                    "temperature": value.get("temperature"), "top_p": value.get("top_p")}
        if stream:
            self.open_stream()
            self.response_sequence = 0
            self.response_event("response.created", response=response)
            self.response_event("response.in_progress", response=response)
            self.response_event("response.output_item.added", output_index=0, item=item)
            self.response_event("response.content_part.added", **position, part=part)
        text, collected = [], 0
        for event in self.generation_events(first, request, deadline):
            if piece := event.get("text"):
                collected += len(json_bytes(piece)) - 2
                self.check_response_size(collected)
                text.append(piece)
                if stream:
                    self.response_event("response.output_text.delta", **position, delta=piece, logprobs=[])
        part["text"] = "".join(text)
        complete = event["reason"] != "length"
        status = "completed" if complete else "incomplete"
        input_tokens, output_tokens = event["prompt_tokens"], event["emitted"]
        item.update(status=status, content=[part])
        response.update(status=status, output=[item], usage={
            "input_tokens": input_tokens,
            "input_tokens_details": {"cached_tokens": event.get("cached_prompt_tokens", 0)},
            "output_tokens": output_tokens, "output_tokens_details": {"reasoning_tokens": 0},
            "total_tokens": input_tokens + output_tokens,
        })
        if complete:
            response["completed_at"] = int(time.time())
        else:
            response["incomplete_details"] = {"reason": "max_output_tokens"}
        if stream:
            self.response_event("response.output_text.done", **position, text=part["text"], logprobs=[])
            self.response_event("response.content_part.done", **position, part=part)
            self.response_event("response.output_item.done", output_index=0, item=item)
            self.response_event(f"response.{status}", response=response)
        else:
            self.inference_reply(response)

    def generation(self, first, request, deadline, kind, stream, include_usage, wants_logprobs, tool_name):
        prefix = "chatcmpl" if kind == "chat" else "cmpl"
        identity = {"id": f"{prefix}-{uuid4().hex}", "created": int(time.time()), "model": self.model_id}
        object_type = "chat.completion.chunk" if kind == "chat" and stream else (
            "chat.completion" if kind == "chat" else "text_completion")

        def envelope(choices, **extra):
            return dict(identity, object=object_type, choices=choices, **extra)

        call_id = f"call_{uuid4().hex}" if tool_name is not None else None

        def tool_call(arguments):
            return {"id": call_id, "type": "function", "function": {"name": tool_name, "arguments": arguments}}

        token_ids, text, scores = [], [], []
        collected = 0
        if stream:
            self.open_stream()
            if kind == "chat":
                delta = {"role": "assistant"}
                if tool_name is not None:
                    delta["tool_calls"] = [dict(tool_call(""), index=0)]
                self.sse(envelope([{"index": 0, "delta": delta, "finish_reason": None}]))
        for event in self.generation_events(first, request, deadline):
            piece = event.get("text", "")
            score = event.get("logprobs")
            if score is not None and kind == "chat":
                score = chat_logprob(score)
            terminal = event["event"] == "done"
            reason = "length" if event.get("reason") == "length" else "stop"
            if terminal and tool_name is not None and reason == "stop":
                reason = "tool_calls"
            if stream:
                if kind == "chat":
                    if piece or score is not None:
                        delta = {}
                        if piece:
                            delta = ({"content": piece} if tool_name is None else
                                     {"tool_calls": [{"index": 0, "function": {"arguments": piece}}]})
                        chunk = {"index": 0, "delta": delta, "finish_reason": None}
                        if score is not None:
                            chunk["logprobs"] = {"content": [score], "refusal": None}
                        self.sse(envelope([chunk]))
                    if terminal:
                        self.sse(envelope([{"index": 0, "delta": {}, "finish_reason": reason}]))
                elif kind == "completion":
                    if piece or terminal:
                        self.sse(envelope([{"index": 0, "text": piece, "logprobs": None,
                                            "finish_reason": reason if terminal else None}]))
                else:
                    self.sse(event)
            else:
                if score is not None:
                    scores.append(score)
                    collected += len(json_bytes(score)) + 1
                if piece:
                    text.append(piece)
                    collected += len(json_bytes(piece)) - 2
                if kind == "infer" and "token_id" in event:
                    token_ids.append(event["token_id"])
                    collected += len(str(event["token_id"])) + 1
                self.check_response_size(collected)
        if stream:
            if include_usage:
                self.sse(envelope([], usage=completion_usage(event)))
            self.sse("[DONE]")
        else:
            result = "".join(text)
            if kind == "infer":
                result = dict(event, generated_ids=token_ids, text=result)
                if wants_logprobs:
                    result["token_logprobs"] = scores
                self.inference_reply(result)
                return
            choice = {"index": 0, "finish_reason": reason}
            if kind == "chat":
                message = {"role": "assistant", "content": result}
                if reason == "tool_calls":
                    message.update(content=None, tool_calls=[tool_call(result)])
                choice["message"] = message
                if wants_logprobs:
                    choice["logprobs"] = {"content": scores, "refusal": None}
            else:
                choice.update(text=result, logprobs=None)
            self.inference_reply(envelope([choice], usage=completion_usage(event)))

    def encoder_results(self, output, width, inputs, deadline):
        # Leave room for a burst of raw JSON floats even when the HTTP reply uses base64.
        frame_bytes = 32 * width + 256
        count = max(1, (self.server.options.max_buffer_bytes - 128) // frame_bytes)
        for start in range(0, len(inputs), count):
            group = inputs[start:start + count]
            multiple = len(group) > 1
            payload = {"output": output, "inputs": group} if multiple else dict(group[0], output=output)
            self.check_peer()
            if time.monotonic() >= deadline:
                raise RequestError("inference deadline exceeded", 504)
            request = self.worker.submit(payload)
            event = "tensor" if multiple else "result"
            try:
                for index in range(len(group)):
                    result = self.waiting(request, deadline)
                    if result["event"] != event or (multiple and result.get("index") != index):
                        raise RequestError("invalid encoder worker response", 502)
                    yield result
                if multiple and self.waiting(request, deadline).get("reason") != "result":
                    raise RequestError("missing encoder worker result", 502)
            finally:
                self.worker.cancel(request)

    def embeddings(self, value, deadline):
        check_fields(value, {"input", "encoding_format", "dimensions", "user"})
        output_name = "sentence_embedding"
        if output_name not in self.worker.outputs or not self.worker.metadata.get("encoder_batch"):
            raise RequestError("checkpoint does not expose sentence_embedding")
        encoding = value.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise RequestError("encoding_format must be float or base64")
        output = next(item for item in self.worker.metadata["outputs"] if item["name"] == output_name)
        dimensions = value.get("dimensions", output["width"])
        if type(dimensions) is not int or dimensions != output["width"]:
            raise RequestError(f"checkpoint has fixed embedding width {output['width']}")

        inputs = value.get("input")
        if isinstance(inputs, str) or token_input(inputs):
            inputs = [inputs]
        elif not (isinstance(inputs, list) and 1 <= len(inputs) <= 16 and
                  (all(isinstance(item, str) for item in inputs) or all(token_input(item) for item in inputs))):
            raise RequestError("expected text, tokens, or a batch of up to 16 inputs of the same type")
        payloads = [prompt_input(item) for item in inputs]
        data = bytearray(b'{"object":"list","model":' + json_bytes(self.model_id) + b',"data":[')
        prompt_tokens = 0
        with closing(self.encoder_results(output_name, output["width"], payloads, deadline)) as results:
            for index, result in enumerate(results):
                vector = result["values"]
                if encoding == "base64":
                    vector = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
                entry = json_bytes({"object": "embedding", "index": index, "embedding": vector})
                self.check_response_size(len(data) + bool(index) + len(entry))
                if index:
                    data.extend(b",")
                data.extend(entry)
                prompt_tokens += result["prompt_tokens"]
        data.extend(b'],"usage":' + json_bytes({"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}) + b"}")
        self.inference_body(data)

    def change_lora(self, load):
        if self.server.options.lora_root is None:
            raise RequestError("runtime adapter management requires --lora-root", 403)
        value = self.body()
        fields = {"lora_name", "lora_path", "base_model_name"} if load else {"lora_name"}
        check_fields(value, fields)
        name = value.get("lora_name")
        if not isinstance(name, str) or not name or name in self.server.pool.slots:
            raise RequestError("invalid or reserved LoRA adapter name")
        if load:
            path = value.get("lora_path")
            if not isinstance(path, str) or not path or "\0" in path:
                raise RequestError("lora_path must be a local adapter directory")
            base = value.get("base_model_name", self.server.default_model)
            if base != self.server.default_model:
                raise RequestError("LoRA adapters attach to the primary model")
            payload = {"load_lora": {"name": name, "path": path}}
        else:
            payload = {"unload_lora": name}
        deadline = time.monotonic() + self.server.options.request_timeout
        with self.server.pool.borrow(self.server.default_model, deadline, self.check_peer) as self.worker:
            request = self.worker.submit(payload)
            try:
                self.waiting(request, deadline)
                self.reply({"lora_name": name, "status": "loaded" if load else "unloaded"})
            finally:
                self.worker.cancel(request)

    def rerank(self, value, deadline):
        check_fields(value, {"query", "documents", "top_n", "user"})
        metadata = self.worker.metadata
        if (not metadata.get("encoder_batch") or not metadata["text"] or
                "token_type_ids" not in metadata["inputs"] or
                not any(output["name"] == "logits" and output["width"] == 1 and not output["sequence"]
                        for output in metadata["outputs"])):
            raise RequestError("reranking requires a text-pair checkpoint with one logit")
        query, documents = value.get("query"), value.get("documents")
        if not isinstance(query, str) or not query:
            raise RequestError("query must be a nonempty string")
        if isinstance(documents, str):
            documents = [documents]
        if (not isinstance(documents, list) or not 1 <= len(documents) <= 16 or
                not all(isinstance(document, str) and document for document in documents)):
            raise RequestError("documents must contain 1 to 16 nonempty strings")
        top_n = value.get("top_n", 0)
        if type(top_n) is not int or top_n < 0:
            raise RequestError("top_n must be a nonnegative integer")
        payloads = [{"prompt": query, "text_pair": document} for document in documents]
        ranked = []
        total_tokens = 0
        with closing(self.encoder_results("logits", 1, payloads, deadline)) as results:
            for index, result in enumerate(results):
                ranked.append((index, result["values"][0]))
                total_tokens += result["prompt_tokens"]
        ranked.sort(key=lambda pair: pair[1], reverse=True)
        results = []
        for index, logit in ranked[:top_n or len(ranked)]:
            exp = math.exp(-abs(logit))
            score = 1 / (1 + exp) if logit >= 0 else exp / (1 + exp)
            results.append({"index": index, "document": {"text": documents[index]}, "relevance_score": score})
        self.inference_reply({"id": f"rerank-{uuid4().hex}", "model": self.model_id,
                              "usage": {"total_tokens": total_tokens}, "results": results})

    def do_POST(self):
        request = None
        lease = ExitStack()
        admitted = self.server.inference_slots.acquire(blocking=False)
        try:
            if not admitted:
                raise RequestError("request limit reached", 429)
            self.check_host()
            if (origin := self.headers.get("Origin")) is not None and origin != f"http://{self.headers['Host']}":
                raise RequestError("expected a same-origin browser request", 403)
            if self.path in {"/v1/load_lora_adapter", "/v1/unload_lora_adapter"}:
                self.change_lora(self.path == "/v1/load_lora_adapter")
                return
            kind = {"/infer": "infer", "/v1/chat/completions": "chat", "/v1/responses": "responses",
                    "/v1/completions": "completion", "/v1/embeddings": "embeddings",
                    "/rerank": "rerank", "/v1/rerank": "rerank", "/v2/rerank": "rerank"}.get(self.path)
            if kind is None:
                raise RequestError("endpoint not found", 404)
            value = self.body()
            if {"lora", "load_lora", "unload_lora"} & value.keys():
                raise RequestError("select a loaded LoRA adapter with the model field")
            self.model_id = value.pop("model", self.server.default_model)
            deadline = time.monotonic() + self.server.options.request_timeout
            parent, _ = self.server.pool.resolve(self.model_id)
            self.worker = lease.enter_context(self.server.pool.borrow(parent, deadline, self.check_peer))
            if kind == "embeddings":
                self.embeddings(value, deadline)
                return
            if kind == "rerank":
                self.rerank(value, deadline)
                return
            structured, schema, tool_name = False, None, None
            if kind == "chat":
                structured, schema = structured_format(value.pop("response_format", None))
                function = chat_tool(value)
                messages = value.get("messages")
                has_tool_history = isinstance(messages, list) and any(
                    isinstance(message, dict) and (message.get("role") == "tool" or message.get("tool_calls"))
                    for message in messages)
                if (function is not None or has_tool_history) and not self.worker.metadata.get("function_calls"):
                    raise RequestError("function calls require the supported SmolLM2 chat template")
                if "messages" in value:
                    value["messages"] = tool_history(messages)
                if function is not None:
                    if structured:
                        raise RequestError("choose either response_format or tool calling")
                    messages = value.get("messages")
                    if (not isinstance(messages, list) or not messages or messages[-1].get("role") != "user" or
                            not isinstance(messages[-1].get("content"), str)):
                        raise RequestError("function calls require a final user message or tool result")
                    tool_name = function["name"]
                    schema = function.get("parameters", {"type": "object", "additionalProperties": False})
                    if not isinstance(schema, dict):
                        raise RequestError("function parameters must be a JSON Schema object")
                    structured = True
                    declaration = json_bytes(function).decode()
                    messages[-1] = dict(messages[-1], content=messages[-1]["content"] +
                                        "\n\nReturn the JSON arguments for this function:\n" + declaration)
            elif kind == "responses" and (text := value.get("text")) is not None:
                if not isinstance(text, dict):
                    raise RequestError("text must be an object")
                check_fields(text, {"format"})
                structured, schema = structured_format(text.get("format"), responses=True)
            payload, input_kind = self.response_input(value) if kind == "responses" else (value, kind)
            stream, tensor, include_usage = self.normalize(payload, input_kind)
            if self.model_id != parent:
                payload["lora"] = self.model_id
            if structured:
                self.apply_schema(payload, schema)
            request = self.worker.submit(payload)
            first = self.waiting(request, deadline)
            if tensor:
                self.inference_reply(first)
            elif kind == "responses":
                self.response_generation(first, request, deadline, stream, value)
            else:
                self.generation(first, request, deadline, kind, stream, include_usage,
                                "top_logprobs" in payload, tool_name)
        except RequestError as error:
            self.send_error_response(error)
        except TimeoutError:
            if not self.started_stream:
                self.send_error_response(RequestError("connection deadline exceeded", 408))
        except (OSError, ConnectionError):
            self.close_connection = True
        except (ValueError, TypeError, OverflowError):
            self.send_error_response(RequestError("invalid request fields"))
        finally:
            self.close_connection = True
            if request is not None:
                self.worker.cancel(request)
            lease.close()
            if admitted:
                self.server.inference_slots.release()


def bounded(low, high):
    def parse(text):
        value = int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"expected a value from {low} to {high}")
        return value
    return parse


def print_generation_stats(event):
    stats = event["stats"]
    fields = [f"{stats['prompt_tokens']} prompt tokens ({stats['cached_prompt_tokens']} cached)",
              f"{stats['output_tokens']} output tokens ({event['reason']})"]
    for key, label, unit in (("ttft_ms", "TTFT", "ms"), ("tpot_ms", "TPOT", "ms/token"),
                             ("prefill_tokens_s", "prefill", "tok/s"), ("decode_tokens_s", "decode", "tok/s"),
                             ("output_tokens_s", "overall", "tok/s"), ("queue_ms", "queue", "ms"),
                             ("elapsed_ms", "total", "ms")):
        if stats[key] is not None:
            fields.append(f"{label} {stats[key]:.2f} {unit}")
    if kv := event.get("kv"):
        fields.append(f"worker KV peak {kv['peak_bytes'] / 2**20:.2f} MiB")
    print("nathr: " + " | ".join(fields), file=sys.stderr, flush=True)


def terminal_input(worker, value, timeout, stats):
    start = time.monotonic()
    request = worker.submit(value)
    deadline = time.monotonic() + timeout
    text = []
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            event = worker.take(request, remaining)
            if event is None:
                break
            event = checked_event(event)
            if event["event"] == "result":
                elapsed = (time.monotonic() - start) * 1000
                print(json_bytes({key: item for key, item in event.items() if key not in ("event", "id")}).decode(),
                      flush=True)
                if stats:
                    print(f"nathr: {event['prompt_tokens']} prompt tokens | request {elapsed:.2f} ms",
                          file=sys.stderr, flush=True)
                return
            if piece := event.get("text"):
                text.append(piece)
                print(piece, end="", flush=True)
            if event["event"] == "done":
                print(flush=True)
                if stats:
                    print_generation_stats(event)
                return "".join(text)
        raise RequestError("inference deadline exceeded")
    except (RequestError, KeyboardInterrupt):
        if text:
            print(flush=True)
        raise
    finally:
        worker.cancel(request)


def run_terminal(worker, options):
    metadata = worker.metadata
    if not metadata["text"]:
        raise RequestError("this checkpoint needs a tokenizer for text input. Use token IDs with nathr infer or generate")
    outputs = [item["name"] for item in metadata["outputs"]]
    output = options.output
    if output is None and not metadata["generation"]:
        output = next((name for name in ("sentence_embedding", "logits", *outputs) if name in outputs), None)
        if output is None:
            raise RequestError("checkpoint has no available text outputs")
    if output is not None and output not in outputs:
        raise RequestError(f"unknown output {output!r}. Available outputs: {', '.join(outputs)}")
    if output is not None and options.max_new_tokens is not None:
        raise RequestError("--max-new-tokens requires text generation")
    count = options.max_new_tokens
    if count is not None and count > options.max_output_tokens:
        raise RequestError("--max-new-tokens exceeds --max-output-tokens")

    history = []
    chat = output is None and metadata["chat"]

    def turn(prompt):
        if output is not None:
            value = {"prompt": prompt, "output": output}
        elif chat:
            messages = [*history, {"role": "user", "content": prompt}]
            value = {"messages": messages}
        else:
            value = {"prompt": prompt}
        if count is not None:
            value["max_new_tokens"] = count
        if options.lora:
            value["lora"] = "nathr-lora"
        reply = terminal_input(worker, value, options.request_timeout, options.stats)
        if chat:
            history.extend((messages[-1], {"role": "assistant", "content": reply}))

    if options.prompt is not None:
        turn(options.prompt)
        return
    interactive = sys.stdin.isatty()
    if interactive:
        commands = "/reset to clear the conversation. /exit to quit." if chat else "/exit to quit."
        print(f"{options.model_id}. {commands}", file=sys.stderr)
    while True:
        try:
            line = input(">>> " if interactive else "")
        except EOFError:
            break
        if not line.strip():
            continue
        if line == "/exit":
            break
        if chat and line == "/reset":
            history.clear()
            continue
        try:
            turn(line)
        except RequestError as error:
            if not interactive or worker.error:
                raise
            print(f"nathr: {error}", file=sys.stderr)


def main(executable, directory, mode, invoked, version, args):
    sys.path.insert(0, directory)
    if mode == "update":
        from update import command
        command(executable, directory, invoked, version, args)
        return
    if mode == "bench-serve":
        from bench_serve import benchmark
        benchmark(args)
        return
    from model_store import ModelStore, command as model_command
    from serving_metrics import ModelMetrics

    if mode in ("models", "pull"):
        model_command(mode, args)
        return
    run = mode == "run"
    parser = argparse.ArgumentParser(prog=f"nathr {mode}",
                                     description=("Start a terminal session or run one prompt." if run else
                                                  "Serve the OpenAI-compatible API on 127.0.0.1."),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("model", nargs="?", metavar="MODEL",
                       help="checkpoint name or local path. Uses smollm2 when omitted")
    model.add_argument("--model", dest="model_path", metavar="PATH", help="path to the checkpoint")
    parser.add_argument("--models-dir", help="cache directory for named checkpoints (overrides NATHR_MODELS)")
    if run:
        parser.add_argument("prompt", nargs="?", help="one prompt to run in the terminal")
        parser.add_argument("--prompt", dest="prompt_text", help="run one prompt")
        parser.add_argument("--no-open", action="store_true", help=argparse.SUPPRESS)
        parser.add_argument("--max-new-tokens", type=bounded(0, 8192),
                            help="generated tokens per turn (checkpoint default or up to 1024)")
        parser.add_argument("--output", help="return a named tensor as JSON")
        parser.add_argument("--lora", action="append", default=[], metavar="DIR", help="apply a local PEFT LoRA adapter")
    else:
        parser.add_argument("--also", action="append", default=[], metavar="MODEL",
                            help="load an additional checkpoint under its usual ID; repeat to load more")
        parser.add_argument("--lora", action="append", default=[], metavar="NAME=DIR",
                            help="register a PEFT LoRA adapter for the primary checkpoint; repeat to add more")
        parser.add_argument("--lora-root", metavar="DIR",
                            help="enable runtime loading and unloading of local adapters beneath this directory")
        parser.add_argument("--memory-budget-mib", type=int,
                            help="account total worker memory and load additional models on demand")
    parser.set_defaults(lora_root=None)
    parser.add_argument("--model-id", help=argparse.SUPPRESS if run else "API model ID, otherwise the checkpoint name")
    parser.add_argument("--port", type=bounded(0, 65535), default=0 if run else 8000,
                        help=argparse.SUPPRESS if run else "HTTP port, or 0 for an available port")
    parser.add_argument("--backend", choices=("scalar", "auto", "avx2"), default="auto", help="CPU kernels")
    parser.add_argument("--threads", type=bounded(1, 64), default=1, help="CPU threads")
    parser.add_argument("--tokenizer", help="matching GGUF tokenizer sidecar")
    parser.add_argument("--chat-format", choices=("auto", "none", "chatml", "qwen3", "zephyr"), default="auto",
                        help="chat format for the primary checkpoint")
    parser.add_argument("--stats", action="store_true",
                        help="print inference stats to stderr" if run else "include request stats in /infer replies")

    scheduler = parser.add_argument_group("generation scheduler")
    scheduler.add_argument("--context", type=bounded(1, 65536), help="context, otherwise up to 8192 tokens")
    scheduler.add_argument("--prefill-chunk", type=bounded(1, 32), default=16, help="prompt tokens per request step")
    scheduler.add_argument("--max-active", type=bounded(1, 64), default=4, help="active generation requests")
    scheduler.add_argument("--step-tokens", type=bounded(1, 128), default=32, help="total tokens per scheduler step")
    scheduler.add_argument("--speculative-tokens", type=bounded(0, 8), default=0,
                           help="prompt lookup candidates to verify in greedy decoding")
    scheduler.add_argument("--max-queued", type=bounded(1, 256), default=16, help="queued generation requests")
    scheduler.add_argument("--max-queued-tokens", type=bounded(1, 262144), default=16384, help="queued prompt tokens")
    scheduler.add_argument("--max-queue-bypasses", type=bounded(0, 256), default=0,
                           help="fitting requests that may pass a KV-blocked queue head")
    scheduler.add_argument("--kv-budget-mib", type=bounded(1, 4096), default=1024, help="KV cache budget in MiB")
    scheduler.add_argument("--kv-cache", choices=("f32", "f16"), help="KV cache precision, automatically selected if omitted")
    scheduler.add_argument("--no-prefix-cache", action="store_true", help="disable decoder prefix and T5 source caching")

    limits = parser.add_argument_group("request limits" if run else "HTTP limits")
    limits.add_argument("--max-clients", type=bounded(1, 128), default=16,
                        help=argparse.SUPPRESS if run else "simultaneous inference clients")
    limits.add_argument("--max-body-bytes", type=bounded(1, 262144), default=262144,
                        help=argparse.SUPPRESS if run else "request body size in bytes")
    limits.add_argument("--max-output-tokens", type=bounded(1, 8192), default=4096, help="generated tokens per request")
    limits.add_argument("--max-result-values", type=bounded(1, 262144), default=65536, help="values per named tensor")
    limits.add_argument("--max-buffer-bytes", type=bounded(128, 8388608), default=2097152,
                        help="response buffer and streaming event limit in bytes")
    limits.add_argument("--io-timeout", type=bounded(1, 60), default=5,
                        help=argparse.SUPPRESS if run else "socket I/O timeout in seconds")
    limits.add_argument("--request-timeout", type=bounded(1, 3600), default=600, help="inference timeout in seconds")
    options = parser.parse_args(args)
    budget = getattr(options, "memory_budget_mib", None)
    if budget is not None and budget <= 0:
        parser.error("--memory-budget-mib must be positive")
    if run:
        if len(options.lora) > 1 or options.lora == [""]:
            parser.error("--lora requires one adapter directory")
        options.lora = next(iter(options.lora), None)
        if options.prompt is not None and options.prompt_text is not None:
            parser.error("pass one prompt")
        if options.prompt_text is not None:
            options.prompt = options.prompt_text
        if options.no_open and options.prompt is None and (
                options.max_new_tokens is not None or options.output or options.lora):
            parser.error("terminal options cannot be used with --no-open")
    if options.lora_root == "":
        parser.error("--lora-root requires a directory")
    if options.max_active > options.step_tokens:
        parser.error("--step-tokens must cover all active slots")
    serving = not run or (options.no_open and options.prompt is None)
    store = ModelStore(options.models_dir)
    model_path, model_id, source = store.locate(options.model or options.model_path)
    options.model_id = options.model_id or model_id
    checkpoints = {options.model_id: WorkerSlot(model_path, source)}
    for name in getattr(options, "also", []):
        path, model_id, source = store.locate(name)
        if model_id in checkpoints:
            parser.error(f"duplicate model ID: {model_id}")
        checkpoints[model_id] = WorkerSlot(path, source)
    adapters = ([f"nathr-lora={options.lora}"] if options.lora else []) if run else options.lora
    configured_adapters = {}
    for spec in adapters:
        name, separator, path = spec.partition("=")
        if not (name and separator and path):
            parser.error("--lora requires NAME=DIR")
        if name in checkpoints or name in configured_adapters:
            parser.error(f"duplicate or reserved LoRA adapter name: {name}")
        configured_adapters[name] = path
    checkpoints[options.model_id].adapters = {name: (path, False) for name, path in configured_adapters.items()}
    if sys.stdin.isatty() and sys.stderr.isatty():
        from update import startup_notice
        startup_notice(executable, version)
    worker_args = []
    for name in ("backend", "threads", "context", "prefill_chunk", "max_active", "step_tokens",
                 "speculative_tokens", "max_queued",
                 "max_queued_tokens", "max_queue_bypasses", "kv_budget_mib", "kv_cache", "max_result_values",
                 "max_output_tokens"):
        if (value := getattr(options, name)) is not None:
            worker_args.extend(("--" + name.replace("_", "-"), str(value)))
    if options.no_prefix_cache:
        worker_args.append("--no-prefix-cache")
    if options.stats:
        worker_args.append("--stats")
    if serving:
        worker_args.append("--metrics")
    with ExitStack() as cleanup:
        pool = WorkerPool(executable, store, checkpoints, options, worker_args, ModelMetrics if serving else None)
        cleanup.callback(pool.close)
        preload = (options.model_id,) if pool.budget else checkpoints
        for model_id in preload:
            deadline = time.monotonic() + options.request_timeout if pool.budget else None
            with pool.borrow(model_id, deadline):
                pass
        if not serving:
            with pool.borrow(options.model_id) as worker:
                run_terminal(worker, options)
            return
        with Server(pool, options) as server:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            print(f"Listening on {url}", flush=True)
            server.serve_forever()


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        main(*sys.argv[1:6], sys.argv[6:])
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, RequestError) as error:
        sys.exit(f"nathr: {error}")
