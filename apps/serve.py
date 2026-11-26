"""Loopback HTTP frontend for one persistent Nathr JSONL worker."""

import argparse
import base64
from collections import deque
from dataclasses import dataclass, field
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import select
import shutil
import signal
import socket
import struct
import subprocess
import sys
import threading
import time
from urllib.parse import unquote


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
    return {"prompt_tokens": prompt, "completion_tokens": generated, "total_tokens": prompt + generated}


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


class Worker:
    def __init__(self, command, limits):
        self.limits = limits
        self.changed = threading.Condition()
        self.pending = {}
        self.next_id = 0
        self.cancelled = 0
        self.metadata = None
        self.error = None
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE, bufsize=0)
        # Buffered reads keep the pipe efficient and cap each worker message.
        self.stdout = open(self.process.stdout.fileno(), "rb", closefd=False)
        self.threads = [threading.Thread(target=work, daemon=True) for work in (self.read, self.write)]
        for thread in self.threads:
            thread.start()

    def start(self):
        with self.changed:
            if not self.changed.wait_for(lambda: self.metadata is not None or self.error, timeout=60):
                raise RuntimeError("model did not become ready")
            if self.error:
                raise self.error

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
            request = Request(self.next_id, json_bytes(dict(value, id=self.next_id)) + b"\n")
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
            cap = 64 * self.limits.max_result_values + 4096
            while line := self.stdout.readline(cap + 1):
                if len(line) > cap or not line.endswith(b"\n"):
                    raise RuntimeError("worker response exceeds its frame limit")
                event = read_json(line)
                with self.changed:
                    if event["event"] == "ready":
                        self.metadata = event
                        self.changed.notify_all()
                        continue
                    request = self.pending.get(event.get("id"))
                    if request is None:
                        continue
                    terminal = event["event"] in {"done", "error", "result"}
                    if terminal:
                        del self.pending[request.id]
                        self.cancelled += event.get("reason") == "cancelled"
                    if request.cancelled:
                        continue
                    if request.buffered + len(line) > self.limits.max_buffer_bytes or len(request.events) == 256:
                        request.error = RequestError("response buffer limit reached", 429)
                        self.cancel_locked(request)
                    else:
                        request.events.append((event, len(line)))
                        request.buffered += len(line)
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
            )


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def __init__(self, worker, options):
        directory = Path(__file__).with_name("nathr-web")
        self.assets = {route: ((directory / name).read_bytes(), content_type) for route, name, content_type in (
            ("/", "index.html", "text/html; charset=utf-8"),
            ("/app.css", "app.css", "text/css; charset=utf-8"),
            ("/app.js", "app.js", "text/javascript; charset=utf-8"),
        )}
        super().__init__(("127.0.0.1", options.port), Handler)
        self.worker = worker
        self.options = options
        self.model = {"id": options.model_id, "object": "model", "owned_by": "nathr", "created": 0}
        self.capabilities = {key: value for key, value in worker.metadata.items() if key != "event"}
        self.outputs = {item["name"] for item in self.capabilities["outputs"]}
        self.slots = threading.BoundedSemaphore(options.max_clients)

    def process_request(self, request, client_address):
        if self.slots.acquire(blocking=False):
            try:
                super().process_request(request, client_address)
            except Exception:
                self.slots.release()
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
            self.slots.release()


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.options.io_timeout)
        self.started_stream = False

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
            if self.started_stream:
                self.sse(response)
                self.sse("[DONE]")
            else:
                self.reply(response, error.status)
        except OSError:
            self.close_connection = True

    def do_GET(self):
        try:
            self.check_host()
            if self.path in self.server.assets:
                self.send_body(*self.server.assets[self.path], ui=True)
            elif self.path == "/health":
                health = self.server.worker.health()
                self.reply(health, 503 if health["status"] == "unavailable" else 200)
            elif self.path == "/v1/models":
                self.reply({"object": "list", "data": [self.server.model]})
            elif (self.path.startswith("/v1/models/") and
                  unquote(self.path.removeprefix("/v1/models/")) == self.server.model["id"]):
                self.reply(self.server.model)
            elif self.path == "/model":
                options = self.server.options
                limits = {name: getattr(options, name) for name in
                          ("max_body_bytes", "max_output_tokens", "max_result_values", "max_buffer_bytes")}
                self.reply(dict(self.server.capabilities, id=options.model_id, limits=limits))
            else:
                raise RequestError("endpoint not found", 404)
        except RequestError as error:
            self.send_error_response(error)
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
        options, metadata = self.server.options, self.server.capabilities
        stream = value.pop("stream", False)
        if not isinstance(stream, bool):
            raise RequestError("stream must be a boolean")
        include_usage = False
        if kind != "infer":
            allowed = {"max_tokens", "temperature", "top_p", "stop", "seed", "n", "stream_options", "user"}
            check_fields(value, allowed | ({"messages", "max_completion_tokens"} if kind == "chat" else {"prompt"}))
            value.pop("user", None)
            count = value.pop("n", 1)
            if type(count) is not int or count != 1:
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
            else:
                if not metadata["text"]:
                    raise RequestError("text completions require a tokenizer")
                value.update(prompt_input(value.pop("prompt", None)))
                if "tokens" in value:
                    value["decode"] = True
            if "max_tokens" in value and "max_completion_tokens" in value:
                raise RequestError("choose max_tokens or max_completion_tokens")
            for name in ("max_tokens", "max_completion_tokens"):
                if name in value:
                    value["max_new_tokens"] = value.pop(name)
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
            if not isinstance(value["output"], str) or value["output"] not in self.server.outputs:
                raise RequestError("checkpoint does not expose the requested output")
        else:
            if not metadata["generation"]:
                raise RequestError("checkpoint requires a named output")
            default = min(metadata["max_new_tokens"], options.max_output_tokens)
            tokens = value.setdefault("max_new_tokens", default)
            if type(tokens) is not int or not 0 <= tokens <= options.max_output_tokens:
                raise RequestError("max_new_tokens exceeds the server limit")
        return stream, tensor, include_usage

    def check_peer(self):
        if select.select([self.connection], [], [], 0)[0]:
            if not self.connection.recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                raise ConnectionError("client disconnected")

    def waiting(self, request, deadline):
        while (remaining := deadline - time.monotonic()) > 0:
            self.check_peer()
            if result := self.server.worker.take(request, min(remaining, 0.1)):
                if result["event"] == "error" or result.get("reason") == "error":
                    status = 429 if result.get("code") == "resource_limit" else 400
                    raise RequestError(result.get("message", "inference failed"), status)
                return result
        raise RequestError("inference deadline exceeded", 504)

    def open_stream(self):
        self.close_connection = True
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()
        self.started_stream = True

    def sse(self, value):
        data = b"[DONE]" if value == "[DONE]" else json_bytes(value)
        self.wfile.write(b"data: " + data + b"\n\n")
        self.wfile.flush()

    def generation(self, first, request, deadline, kind, stream, include_usage):
        prefix = "chatcmpl" if kind == "chat" else "cmpl"
        identity = {"id": f"{prefix}-{request.id}", "created": int(time.time()),
                    "model": self.server.options.model_id}
        object_type = "chat.completion.chunk" if kind == "chat" and stream else (
            "chat.completion" if kind == "chat" else "text_completion")

        def envelope(choices, **extra):
            return dict(identity, object=object_type, choices=choices, **extra)

        token_ids, text = [], []
        collected = 0
        event = first
        if stream:
            self.open_stream()
            if kind == "chat":
                self.sse(envelope([{"index": 0, "delta": {"role": "assistant"}, "finish_reason": None}]))
        while True:
            piece = event.get("text", "")
            terminal = event["event"] == "done"
            reason = "length" if event.get("reason") == "length" else "stop"
            if stream:
                if kind == "chat":
                    if piece:
                        self.sse(envelope([{"index": 0, "delta": {"content": piece}, "finish_reason": None}]))
                    if terminal:
                        self.sse(envelope([{"index": 0, "delta": {}, "finish_reason": reason}]))
                elif kind == "completion":
                    if piece or terminal:
                        self.sse(envelope([{"index": 0, "text": piece, "logprobs": None,
                                            "finish_reason": reason if terminal else None}]))
                else:
                    self.sse(event)
            else:
                collected += len(json_bytes(piece)) - 2
                if kind == "infer" and "token_id" in event:
                    token_ids.append(event["token_id"])
                    collected += len(str(event["token_id"])) + 1
                if collected > self.server.options.max_buffer_bytes:
                    raise RequestError("response buffer limit reached", 429)
                if piece:
                    text.append(piece)
            if terminal:
                break
            event = self.waiting(request, deadline)
        if stream:
            if include_usage:
                self.sse(envelope([], usage=completion_usage(event)))
            self.sse("[DONE]")
        else:
            result = "".join(text)
            if kind == "infer":
                self.reply(dict(event, generated_ids=token_ids, text=result))
                return
            choice = {"index": 0, "finish_reason": reason}
            if kind == "chat":
                choice["message"] = {"role": "assistant", "content": result}
            else:
                choice.update(text=result, logprobs=None)
            self.reply(envelope([choice], usage=completion_usage(event)))

    def embeddings(self, value, deadline):
        check_fields(value, {"input", "encoding_format", "dimensions", "user"})
        output_name = "sentence_embedding"
        if output_name not in self.server.outputs:
            raise RequestError("checkpoint does not expose sentence_embedding")
        encoding = value.get("encoding_format", "float")
        if encoding not in ("float", "base64"):
            raise RequestError("encoding_format must be float or base64")
        output = next(item for item in self.server.capabilities["outputs"] if item["name"] == output_name)
        dimensions = value.get("dimensions", output["width"])
        if type(dimensions) is not int or dimensions != output["width"]:
            raise RequestError(f"checkpoint has fixed embedding width {output['width']}")

        inputs = value.get("input")
        if isinstance(inputs, str) or token_input(inputs):
            inputs = [inputs]
        elif not (isinstance(inputs, list) and 1 <= len(inputs) <= 16 and
                  (all(isinstance(item, str) for item in inputs) or all(token_input(item) for item in inputs))):
            raise RequestError("expected text, tokens, or a batch of up to 16 inputs of the same type")
        payloads = [dict(prompt_input(item), output=output_name) for item in inputs]
        data, prompt_tokens, collected = [], 0, 0
        worker = self.server.worker
        for index, payload in enumerate(payloads):
            self.check_peer()
            if time.monotonic() >= deadline:
                raise RequestError("inference deadline exceeded", 504)
            request = worker.submit(payload)
            try:
                result = self.waiting(request, deadline)
            finally:
                worker.cancel(request)
            vector = result["values"]
            if encoding == "base64":
                vector = base64.b64encode(struct.pack(f"<{len(vector)}f", *vector)).decode("ascii")
            entry = {"object": "embedding", "index": index, "embedding": vector}
            collected += len(json_bytes(entry)) + 1
            if collected > self.server.options.max_buffer_bytes:
                raise RequestError("response buffer limit reached", 429)
            data.append(entry)
            prompt_tokens += result["prompt_tokens"]
        self.reply({"object": "list", "model": self.server.model["id"], "data": data,
                    "usage": {"prompt_tokens": prompt_tokens, "total_tokens": prompt_tokens}})

    def do_POST(self):
        request = None
        try:
            self.check_host()
            if (origin := self.headers.get("Origin")) is not None and origin != f"http://{self.headers['Host']}":
                raise RequestError("expected a same-origin browser request", 403)
            kind = {"/infer": "infer", "/v1/chat/completions": "chat",
                    "/v1/completions": "completion", "/v1/embeddings": "embeddings"}.get(self.path)
            if kind is None:
                raise RequestError("endpoint not found", 404)
            value = self.body()
            if value.pop("model", self.server.model["id"]) != self.server.model["id"]:
                raise RequestError("model is not loaded", 404)
            deadline = time.monotonic() + self.server.options.request_timeout
            if kind == "embeddings":
                self.embeddings(value, deadline)
                return
            stream, tensor, include_usage = self.normalize(value, kind)
            request = self.server.worker.submit(value)
            first = self.waiting(request, deadline)
            if tensor:
                self.reply(first)
            else:
                self.generation(first, request, deadline, kind, stream, include_usage)
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
                self.server.worker.cancel(request)


def bounded(low, high):
    def parse(text):
        value = int(text)
        if not low <= value <= high:
            raise argparse.ArgumentTypeError(f"expected a value from {low} to {high}")
        return value
    return parse


def run_prompt(worker, options):
    metadata = worker.metadata
    if not metadata["generation"]:
        raise RequestError("this checkpoint returns named tensors. Run without a prompt to use the browser")
    if not metadata["text"]:
        raise RequestError("this checkpoint needs a tokenizer for text input")
    maximum = min(metadata["max_new_tokens"], options.max_output_tokens)
    count = options.max_new_tokens if options.max_new_tokens is not None else maximum
    if count > options.max_output_tokens:
        raise RequestError("--max-new-tokens exceeds --max-output-tokens")
    value = {"max_new_tokens": count}
    if metadata["chat"]:
        value["messages"] = [{"role": "user", "content": options.prompt}]
    else:
        value["prompt"] = options.prompt
    request = worker.submit(value)
    deadline = time.monotonic() + options.request_timeout
    try:
        while (remaining := deadline - time.monotonic()) > 0:
            event = worker.take(request, remaining)
            if event is None:
                break
            if event["event"] == "error" or event.get("reason") == "error":
                raise RequestError(event.get("message", "inference failed"))
            if text := event.get("text"):
                print(text, end="", flush=True)
            if event["event"] == "done":
                print()
                return
        raise RequestError("inference deadline exceeded")
    finally:
        worker.cancel(request)


def start_browser(url):
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        if launcher := shutil.which("xdg-open"):
            try:
                subprocess.Popen([launcher, url], start_new_session=True, stdin=subprocess.DEVNULL,
                                 stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except OSError as error:
                print(f"Could not open the browser: {error}", file=sys.stderr)


def main(executable, directory, mode, args):
    sys.path.insert(0, directory)
    from model_store import ModelStore, command as model_command

    if mode in ("models", "pull"):
        model_command(mode, args)
        return
    run = mode == "run"
    parser = argparse.ArgumentParser(prog=f"nathr {mode}",
                                     description=("Open the local browser app or run one prompt." if run else
                                                  "Start the browser interface and HTTP API on 127.0.0.1."),
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter, allow_abbrev=False)
    model = parser.add_mutually_exclusive_group()
    model.add_argument("model", nargs="?", metavar="MODEL",
                       help="checkpoint name or local path. Uses smollm2 when omitted")
    model.add_argument("--model", dest="model_path", metavar="PATH", help="path to the checkpoint")
    parser.add_argument("--models-dir", help="cache directory for named checkpoints (overrides NATHR_MODELS)")
    if run:
        parser.add_argument("prompt", nargs="?", help="one prompt to run in the terminal")
        parser.add_argument("--prompt", dest="prompt_text", help="run a prompt with the default model")
        parser.add_argument("--no-open", action="store_true", help="print the browser address without opening it")
        parser.add_argument("--max-new-tokens", type=bounded(0, 8192), help="limit tokens for a terminal prompt")
    parser.add_argument("--model-id", help="API model ID, otherwise the checkpoint filename or directory name")
    parser.add_argument("--port", type=bounded(0, 65535), default=0 if run else 8000,
                        help="HTTP port, or 0 for an available port")
    parser.add_argument("--backend", choices=("scalar", "auto", "avx2"), default="auto", help="CPU kernels")
    parser.add_argument("--threads", type=bounded(1, 64), default=1, help="CPU threads")
    parser.add_argument("--tokenizer", help="matching GGUF tokenizer sidecar")

    scheduler = parser.add_argument_group("generation scheduler")
    scheduler.add_argument("--context", type=bounded(1, 65536), help="context, otherwise up to 2048 tokens")
    scheduler.add_argument("--prefill-chunk", type=bounded(1, 32), default=16, help="prompt tokens per request step")
    scheduler.add_argument("--max-active", type=bounded(1, 64), default=4, help="active generation requests")
    scheduler.add_argument("--step-tokens", type=bounded(1, 128), default=32, help="total tokens per scheduler step")
    scheduler.add_argument("--max-queued", type=bounded(1, 256), default=16, help="queued generation requests")
    scheduler.add_argument("--max-queued-tokens", type=bounded(1, 262144), default=16384, help="queued prompt tokens")
    scheduler.add_argument("--kv-budget-mib", type=bounded(1, 4096), default=512, help="KV cache budget in MiB")
    scheduler.add_argument("--kv-cache", choices=("f32", "f16"), default="f32", help="KV cache precision")

    limits = parser.add_argument_group("HTTP limits")
    limits.add_argument("--max-clients", type=bounded(1, 128), default=16, help="simultaneous clients")
    limits.add_argument("--max-body-bytes", type=bounded(1, 262144), default=65536, help="request body size in bytes")
    limits.add_argument("--max-output-tokens", type=bounded(1, 8192), default=256, help="generated tokens per request")
    limits.add_argument("--max-result-values", type=bounded(1, 131072), default=65536, help="values per named tensor")
    limits.add_argument("--max-buffer-bytes", type=bounded(128, 8388608), default=2097152,
                        help="buffer size per response in bytes")
    limits.add_argument("--io-timeout", type=bounded(1, 60), default=5, help="socket I/O timeout in seconds")
    limits.add_argument("--request-timeout", type=bounded(1, 600), default=120, help="inference timeout in seconds")
    options = parser.parse_args(args)
    if run:
        if options.prompt is not None and options.prompt_text is not None:
            parser.error("pass one prompt")
        if options.prompt_text is not None:
            options.prompt = options.prompt_text
        if options.max_new_tokens is not None and options.prompt is None:
            parser.error("--max-new-tokens requires a terminal prompt")
    if options.max_active > options.step_tokens:
        parser.error("--step-tokens must cover all active slots")
    model_path, model_id = ModelStore(options.models_dir).resolve(options.model or options.model_path)
    options.model_id = options.model_id or model_id
    command = [executable, "batch", "--model", str(model_path), "--ready"]
    for name in ("backend", "threads", "context", "prefill_chunk", "max_active", "step_tokens", "max_queued",
                 "max_queued_tokens", "kv_budget_mib", "kv_cache", "tokenizer", "max_result_values"):
        if (value := getattr(options, name)) is not None:
            command.extend(("--" + name.replace("_", "-"), str(value)))
    worker = Worker(command, options)
    try:
        worker.start()
        if run and options.prompt is not None:
            run_prompt(worker, options)
            return
        with Server(worker, options) as server:
            url = f"http://127.0.0.1:{server.server_address[1]}"
            print(f"Listening on {url}", flush=True)
            if run and not options.no_open:
                start_browser(url)
            server.serve_forever()
    finally:
        worker.close()


if __name__ == "__main__":
    def terminate(_signum, _frame):
        raise KeyboardInterrupt
    signal.signal(signal.SIGTERM, terminate)
    try:
        main(*sys.argv[1:4], sys.argv[4:])
    except KeyboardInterrupt:
        pass
    except (OSError, RuntimeError, ValueError, RequestError) as error:
        sys.exit(f"nathr: {error}")
