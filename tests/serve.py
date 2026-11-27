"""Exercise the HTTP frontend and a stalled output consumer with tiny model fixtures."""

import base64
from contextlib import contextmanager
import http.client
import json
import math
from pathlib import Path
import runpy
import select
import socket
import struct
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace


def until(predicate, message):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(0.02)
    raise AssertionError(message)


@contextmanager
def server(binary, checkpoint, *options, named_model=False):
    model = ["--model", checkpoint] if named_model else [checkpoint]
    with subprocess.Popen([binary, "serve", *model, "--port", "0", *options],
                          stdout=subprocess.PIPE, text=True) as process:
        try:
            assert select.select([process.stdout], [], [], 15)[0], "server never became ready"
            ready = process.stdout.readline()
            assert ready.startswith("Listening on http://127.0.0.1:"), ready
            yield int(ready.rsplit(":", 1)[1])
        finally:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()


def request(port, path, value=None, status=200):
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    try:
        body = None if value is None else json.dumps(value, separators=(",", ":"))
        connection.request("GET" if value is None else "POST", path, body)
        response = connection.getresponse()
        raw = response.read()
        assert response.status == status, (path, response.status, raw)
        content_type = response.getheader("Content-Type")
        if content_type == "text/event-stream":
            if path == "/v1/responses":
                events = []
                for frame in raw.rstrip(b"\n").split(b"\n\n"):
                    name, data = frame.split(b"\n")
                    event = json.loads(data.removeprefix(b"data: "))
                    assert name == b"event: " + event["type"].encode()
                    assert event["sequence_number"] == len(events)
                    events.append(event)
                return events
            lines = [line for line in raw.splitlines() if line]
            assert lines[-1] == b"data: [DONE]", lines
            return [json.loads(line.removeprefix(b"data: ")) for line in lines[:-1]]
        if content_type.startswith("text/"):
            assert response.getheader("Content-Security-Policy")
            return raw.decode()
        return json.loads(raw)
    finally:
        connection.close()


def chat_fixture(target, source):
    (target / "model.safetensors").symlink_to(source / "model.safetensors")
    config = json.loads((source / "config.json").read_text())
    (target / "config.json").write_text(json.dumps(dict(config, eos_token_id=[0, 22])))
    # Roles are single tokens so the test can focus on rendering and streaming.
    tokens = ["<|im_end|>", "<|im_start|>", "system", "user", "b", "assistant", "Ċ", "Ġ",
              *"cdefghijklmn", "a", "o", "p"]
    byte_level = dict(type="ByteLevel", add_prefix_space=False)
    added = [dict(id=index, content=tokens[index], special=True, normalized=False,
                  single_word=False, lstrip=False, rstrip=False) for index in (0, 1, 2, 3, 5)]
    tokenizer = dict(version="1.0", normalizer=None, truncation=None, padding=None, post_processor=None,
                     pre_tokenizer=byte_level, decoder=byte_level, added_tokens=added,
                     model=dict(type="BPE", vocab={token: i for i, token in enumerate(tokens)}, merges=[],
                                dropout=None, unk_token=None, continuing_subword_prefix=None,
                                end_of_word_suffix=None))
    (target / "tokenizer.json").write_text(json.dumps(tokenizer))
    template = (
        "{% for message in messages %}{% if loop.first and messages[0]['role'] != 'system' %}"
        "{{ '<|im_start|>system\na<|im_end|>\n' }}{% endif %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}{% if add_generation_prompt %}{{ '<|im_start|>assistant\n' }}{% endif %}"
    )
    (target / "tokenizer_config.json").write_text(json.dumps(dict(chat_template=template)))
    (target / "generation_config.json").write_text(json.dumps(dict(eos_token_id=[0, 22], do_sample=False)))


def http_checks(binary, checkpoint, encoder):
    chat_path = "/v1/chat/completions"
    with server(binary, checkpoint, "--model-id", "test", "--context", "8192", "--max-body-bytes", "16384",
                "--max-queued-tokens", "6100", "--max-result-values", "64", "--step-tokens", "4") as port:
        assert request(port, "/health")["status"] == "ok"
        assert "<title>Nathr</title>" in request(port, "/")
        assert ".workspace" in request(port, "/app.css")
        assert 'fetch("/model")' in request(port, "/app.js")
        request(port, "/web/unknown", status=404)
        model = request(port, "/v1/models")["data"][0]
        assert model["id"] == "test" and request(port, "/v1/models/test") == model
        request(port, "/v1/models/unknown", status=404)
        metadata = request(port, "/model")
        assert metadata["architecture"] == "llama" and metadata["chat"] and metadata["text"]
        assert metadata["generation_context"] == 8192
        assert metadata["outputs"] == [{"name": "logits", "width": 23, "input_limit": 128, "sequence": True}]
        assert metadata["limits"]["max_result_values"] == 64
        chat = dict(model="test", messages=[dict(role="user", content="b")], temperature=0, max_tokens=4)
        complete = request(port, chat_path, chat)
        nullable = request(port, chat_path, dict(chat, max_tokens=None, max_completion_tokens=4, n=None, stream=None))
        assert nullable["choices"] == complete["choices"] and nullable["usage"] == complete["usage"]
        events = request(port, chat_path, dict(chat, stream=True))
        deltas = [event["choices"][0]["delta"] for event in events]
        assert deltas[0] == {"role": "assistant"}
        assert "".join(delta.get("content", "") for delta in deltas) == complete["choices"][0]["message"]["content"]
        assert events[-1]["choices"][0]["finish_reason"] == complete["choices"][0]["finish_reason"]
        assert complete["usage"]["prompt_tokens"] == 15
        with_usage = request(port, chat_path, dict(chat, stream=True, stream_options=dict(include_usage=True)))
        assert with_usage[-1]["choices"] == [] and with_usage[-1]["usage"] == complete["usage"]
        request(port, chat_path, dict(chat, model="other"), 404)
        request(port, chat_path, dict(chat, messages=[dict(role="assistant", content="b")]), 400)

        responses = "/v1/responses"
        payload = dict(model="test", input="b", temperature=0, max_output_tokens=4, store=False)
        response = request(port, responses, payload)
        part = response["output"][0]["content"][0]
        assert response["object"] == "response" and response["status"] == "incomplete"
        assert response["incomplete_details"] == {"reason": "max_output_tokens"}
        assert part["type"] == "output_text" and part["text"] == complete["choices"][0]["message"]["content"]
        assert response["usage"]["input_tokens"] == 15 and response["usage"]["output_tokens"] == 4
        messages = [dict(role="user", content=[dict(type="input_text", text="b")])]
        events = request(port, responses, dict(payload, input=messages, stream=True))
        assert events[0]["type"] == "response.created"
        assert "".join(event["delta"] for event in events if event["type"] == "response.output_text.delta") == part["text"]
        assert events[-1]["type"] == "response.incomplete" and events[-1]["response"]["usage"] == response["usage"]
        request(port, responses, dict(payload, previous_response_id=response["id"]), 400)
        request(port, responses, dict(payload, store=True), 400)
        request(port, responses, dict(payload, input=[dict(role="user", content=[dict(type="input_image")])]), 400)
        request(port, "/infer", dict(prompt="b" * 16400), 413)
        request(port, "/infer", dict(tokens=[4] * 6101, max_new_tokens=1), 429)
        assert request(port, "/infer", dict(tokens=[1, 7], output="logits"))["shape"] == [2, 23]
        request(port, "/infer", dict(tokens=[1, 7, 2], output="logits"), 429)

        tokens = dict(tokens=[4, 4, 4, 4], max_new_tokens=4)
        completion = dict(model="test", prompt="bbbb", temperature=0, max_tokens=4)
        result = request(port, "/v1/completions", completion)
        assert result["object"] == "text_completion" and result["choices"][0]["text"] == "aaaa"
        assert result["usage"] == {"prompt_tokens": 4, "completion_tokens": 4, "total_tokens": 8}
        chunks = request(port, "/v1/completions", dict(completion, prompt=tokens["tokens"], stream=True))
        assert "".join(event["choices"][0]["text"] for event in chunks) == "aaaa"
        assert chunks[-1]["choices"][0]["finish_reason"] == "length"
        request(port, "/v1/completions", dict(completion, prompt=["a", "b"]), 400)
        request(port, "/v1/embeddings", dict(input="b"), 400)
        ids = request(port, "/infer", dict(tokens, stop_ids=[22, 20], decode=True))
        assert ids["generated_ids"] == [20] and ids["text"] == "" and ids["reason"] == "stop"
        stopped = request(port, "/infer", dict(tokens, stop=["aaa"], stream=True))
        assert len([event for event in stopped if event["event"] == "token"]) == 3
        assert stopped[-1]["reason"] == "stop" and not any(event.get("text") for event in stopped)

        stalled = socket.create_connection(("127.0.0.1", port))
        try:
            data = json.dumps(dict(tokens=[4] * 6000, max_new_tokens=100, stream=True), separators=(",", ":")).encode()
            stalled.sendall((f"POST /infer HTTP/1.1\r\nHost: localhost:{port}\r\n"
                             f"Content-Length: {len(data)}\r\n\r\n").encode() + data)
            until(lambda: request(port, "/health")["pending_requests"] == 1, "long request did not enter worker")
            short = request(port, "/infer", tokens)
            assert short["emitted"] == 4
            assert request(port, "/health")["pending_requests"] == 1, "short request waited behind long prefill"
        finally:
            stalled.close()
        until(lambda: request(port, "/health")["cancelled_requests"] == 1, "disconnect did not cancel generation")
        assert request(port, "/health")["pending_requests"] == 0

    with server(binary, encoder) as port:
        result = request(port, "/infer", dict(prompt="The sky is blue.", output="sentence_embedding"))
        assert result["shape"] == [12] and len(result["values"]) == 12
        request(port, "/infer", dict(tokens=[1]), 400)
        metadata = request(port, "/model")
        assert metadata["architecture"] == "bert" and metadata["text"] and not metadata["generation"]
        assert metadata["inputs"] == ["tokens", "token_type_ids"]
        embeddings = "/v1/embeddings"
        texts = ["The sky is blue.", "The sky."]
        expected = [request(port, "/infer", dict(prompt=text, output="sentence_embedding")) for text in texts]
        batch = request(port, embeddings, dict(input=texts))
        assert batch["object"] == "list" and len(batch["data"]) == len(texts)
        assert [item["index"] for item in batch["data"]] == [0, 1]
        assert [item["embedding"] for item in batch["data"]] == [item["values"] for item in expected]
        assert batch["usage"]["total_tokens"] == sum(item["prompt_tokens"] for item in expected)
        token_batch = [[1, 2, 3], [4, 5]]
        encoded = request(port, embeddings, dict(input=token_batch, encoding_format="base64"))
        assert encoded["usage"] == {"prompt_tokens": 5, "total_tokens": 5}
        for ids, item in zip(token_batch, encoded["data"], strict=True):
            vector = request(port, "/infer", dict(tokens=ids, output="sentence_embedding"))["values"]
            assert base64.b64decode(item["embedding"]) == struct.pack("<12f", *vector)
        request(port, embeddings, dict(input=["The sky.", [1, 2]]), 400)
        request(port, embeddings, dict(input=["x"] * 17), 400)
        request(port, embeddings, dict(input="The sky.", dimensions=1), 400)
    with server(binary, encoder, "--max-buffer-bytes", "560", named_model=True) as port:
        request(port, "/v1/embeddings", dict(input="The sky is blue."))
        request(port, "/v1/embeddings", dict(input=["The sky is blue."] * 4), 429)
        compact = request(port, "/v1/embeddings", dict(input=["The sky is blue."] * 4, encoding_format="base64"))
        assert len(compact["data"]) == 4 and len(base64.b64decode(compact["data"][0]["embedding"])) == 48

    with server(binary, checkpoint, "--max-buffer-bytes", "512", "--model-id", "test") as port:
        request(port, "/v1/chat/completions", dict(messages=[dict(role="user", content="b")], max_tokens=1))
        request(port, "/v1/responses", dict(input="b", max_output_tokens=1), 429)
        events = request(port, "/v1/responses", dict(input="b", max_output_tokens=1, stream=True))
        assert events[0]["type"] == "response.created"
        assert events[-1]["type"] == "error" and events[-1]["error"]["code"] == 429
        assert all(len(json.dumps(event, ensure_ascii=False, separators=(",", ":")).encode()) <= 512
                   for event in events)
        assert request(port, "/health")["pending_requests"] == 0

    with server(binary, encoder.parent / "model/f32") as port:
        assert not request(port, "/model")["text"]
        assert request(port, "/infer", dict(tokens=[4, 4], max_new_tokens=1))["emitted"] == 1


def stalled_consumer(binary, checkpoint):
    worker_type = runpy.run_path(str(Path(binary).with_name("nathr-serve.py")))["Worker"]
    limits = SimpleNamespace(max_clients=2, max_result_values=64, max_buffer_bytes=512)
    worker = worker_type([binary, "batch", checkpoint, "--ready", "--max-active", "2", "--step-tokens", "4"], limits)
    try:
        worker.start()
        slow = worker.submit(dict(tokens=[4] * 4, max_new_tokens=512))
        until(lambda: slow.error is not None, "stalled consumer did not reach its buffer limit")
        assert slow.error.status == 429 and slow.buffered <= limits.max_buffer_bytes
        until(lambda: worker.health()["cancelled_requests"] == 1, "overloaded stream did not release its slot")
        short = worker.submit(dict(tokens=[4] * 4, max_new_tokens=1))
        assert worker.take(short, 5)["event"] == "token"
        assert worker.take(short, 5)["reason"] == "length"
    finally:
        worker.close()


def checkpoint_defaults(binary, checkpoint):
    metadata = checkpoint / "generation_config.json"
    command = [binary, "generate", checkpoint, "--tokens", "4,4,4,4", "--json"]
    defaults = dict(do_sample=True, top_k=1, max_new_tokens=3, eos_token_id=[0, 22])
    metadata.write_text(json.dumps(defaults))
    result = json.loads(subprocess.check_output(command, timeout=10))
    assert result["generated_ids"] == [20, 20, 20] and result["finish_reason"] == "length"
    metadata.write_text(json.dumps(dict(defaults, eos_token_id=[0, 20])))
    result = json.loads(subprocess.check_output(command, timeout=10))
    assert result["generated_ids"] == [20] and result["stopped_at_stop"]


def chat_stdin(binary, checkpoint):
    command = [binary, "generate", checkpoint, "--messages", "-", "--max-new-tokens", "1", "--json"]
    messages = [{"role": "user", "content": "b"}]
    result = json.loads(subprocess.check_output(command, input=json.dumps(messages), text=True, timeout=10))
    assert len(result["prompt_ids"]) == 15 and result["emitted"] == 1


def user_commands(binary, checkpoint, encoder, directory):
    output = subprocess.check_output([binary, "models", "--models-dir", directory], text=True, timeout=10)
    assert "smollm2" in output and "available" in output and "(default)" in output
    result = subprocess.run([binary, "pull", "unknown", "--models-dir", directory],
                            text=True, capture_output=True, timeout=10)
    assert result.returncode != 0 and "nathr models" in result.stderr
    limit = ["--max-new-tokens", "4"]
    messages = [{"role": "user", "content": "bbbb"}]
    generate = [binary, "generate", checkpoint, "--messages", "-", *limit]
    reply = subprocess.check_output([binary, "run", checkpoint, "bbbb", *limit], text=True, timeout=10)
    expected = subprocess.check_output(generate, input=json.dumps(messages), text=True, timeout=10)
    assert reply == expected
    terminal = [binary, "run", checkpoint, *limit, "--context", "36"]
    result = subprocess.run(terminal, input="bbbb\nbbbb\n", text=True, capture_output=True, timeout=10)
    assert result.returncode != 0 and "exceed context" in result.stderr and result.stdout == expected
    reply = subprocess.check_output(terminal, input="bbbb\n/reset\nbbbb\n", text=True, timeout=10)
    assert reply == expected + expected
    output = subprocess.check_output([binary, "run", encoder], input="The sky is blue.\n/exit\n",
                                     text=True, timeout=10)
    assert json.loads(output)["shape"] == [12]


def multi_model_benchmark(binary, checkpoint, encoder):
    with server(binary, checkpoint, "--model-id", "chat", "--also", encoder, "--context", "128") as port:
        models = request(port, "/v1/models")["data"]
        assert [model["id"] for model in models] == ["chat", "bert"]
        assert request(port, "/v1/models/bert") == models[1]
        assert request(port, "/model")["id"] == "chat"
        assert request(port, "/model?model=bert")["outputs"] != request(port, "/model")["outputs"]
        assert set(request(port, "/health")["models"]) == {"chat", "bert"}
        request(port, "/health?model=missing", status=404)
        assert request(port, "/health?model=chat")["kv"]["peak_bytes"] == 0
        assert request(port, "/v1/responses", dict(input="b", max_output_tokens=1))["model"] == "chat"
        embedding = request(port, "/v1/embeddings", dict(model="bert", input="The sky is blue."))
        assert embedding["model"] == "bert" and len(embedding["data"][0]["embedding"]) == 12
        request(port, "/v1/responses", dict(model="bert", input="The sky is blue."), status=400)

        report = subprocess.check_output([binary, "bench", "--serve", "--port", str(port), "--model", "chat",
                                          "--workload", "shared-prefix", "--requests", "4", "--concurrency", "2",
                                          "--warmups", "1", "--prompt-length", "48", "--seed", "43",
                                          "--decode-length", "4"], text=True, timeout=15)
        result = json.loads(report)
        assert result["failed"] == 0 and result["cached_prompt_tokens"] >= 64
        assert result["kv_peak_bytes"] > 0 and result["ttft_ms"]["p95"] > 0 and result["queue_ms"]["p95"] >= 0
        mixed = subprocess.check_output([binary, "bench", "--serve", "--port", str(port), "--model", "chat",
                                         "--workload", "mixed", "--requests", "4", "--concurrency", "2",
                                         "--prompt-length", "16", "--seed", "43", "--decode-length", "4"],
                                        text=True, timeout=15)
        result = json.loads(mixed)
        assert result["short"]["requests"] == 3 and result["long"]["requests"] == 1


def encoder_batch_checks(binary, encoder):
    batch = {"output": "sentence_embedding", "inputs": [{"tokens": [1, 2, 3]}, {"tokens": [4, 5, 6]}]}
    commands = [dict(batch, id=1), {"cancel": 1},
                dict(batch, id=2, inputs=[{"tokens": [99]}]), dict(batch, id=3)]
    stdout = subprocess.check_output([binary, "batch", encoder, "--ready"], text=True, timeout=10,
                                     input="".join(json.dumps(item) + "\n" for item in commands))
    frames = [json.loads(line) for line in stdout.splitlines()]
    assert [e["reason"] for e in frames if e.get("id") == 1] == ["cancelled"]
    assert [e["event"] for e in frames if e.get("id") == 2] == ["error"]
    results = [e for e in frames if e.get("id") == 3]
    assert [e["index"] for e in results if e["event"] == "tensor"] == [0, 1]
    assert results[-1]["reason"] == "result"


def rerank_checks(binary, encoder, target):
    target.mkdir()
    for path in encoder.iterdir():
        if path.name not in ("config.json", "model.safetensors"):
            (target / path.name).symlink_to(path.resolve(), target_is_directory=path.is_dir())
    config = json.loads((encoder / "config.json").read_text())
    (target / "config.json").write_text(json.dumps(dict(config, num_labels=1)))
    original = (encoder / "model.safetensors").read_bytes()
    size = int.from_bytes(original[:8], "little")
    header, data = json.loads(original[8:8 + size]), original[8 + size:]
    tensors = sorted((name for name in header if name != "__metadata__"),
                     key=lambda name: header[name]["data_offsets"])
    weights = bytearray()
    for name in tensors:
        tensor = header[name]
        start, end = tensor["data_offsets"]
        if name in ("classifier.weight", "classifier.bias"):
            end = start + (end - start) // tensor["shape"][0]
            tensor["shape"][0] = 1
        tensor["data_offsets"] = [len(weights), len(weights) + end - start]
        weights.extend(data[start:end])
    encoded = json.dumps(header).encode()
    (target / "model.safetensors").write_bytes(len(encoded).to_bytes(8, "little") + encoded + weights)

    with server(binary, encoder, "--also", target) as port:
        query = "The sky?"
        documents = ["The sky is blue.", "Hello world.", "The sky is blue."]
        scores = [request(port, "/infer", dict(model="reranker", prompt=query, text_pair=text, output="logits"))
                  for text in documents]
        order = sorted(range(len(documents)), key=lambda index: scores[index]["values"][0], reverse=True)
        value = dict(model="reranker", query=query, documents=documents, top_n=2)
        result = request(port, "/v1/rerank", value)
        assert result["model"] == "reranker"
        assert result["usage"]["total_tokens"] == sum(score["prompt_tokens"] for score in scores)
        assert [item["index"] for item in result["results"]] == order[:2]
        for item in result["results"]:
            index = item["index"]
            assert item["document"] == {"text": documents[index]}
            assert math.isclose(item["relevance_score"], 1 / (1 + math.exp(-scores[index]["values"][0])),
                                rel_tol=1e-6)
        request(port, "/v1/rerank", dict(value, model="bert"), status=400)
        request(port, "/v1/rerank", dict(value, documents=documents * 6), status=400)
        request(port, "/v1/rerank", dict(value, top_n=-1), status=400)


if __name__ == "__main__":
    executable, fixtures = map(Path, sys.argv[1:])
    help_text = subprocess.check_output([executable, "help", "serve"], text=True, timeout=10)
    assert "HTTP limits" in help_text and "--model-id" in help_text
    with tempfile.TemporaryDirectory(prefix="nathr-serve-") as directory:
        checkpoint = Path(directory)
        chat_fixture(checkpoint, fixtures / "model/f32")
        http_checks(executable, checkpoint, fixtures / "bert")
        stalled_consumer(executable, checkpoint)
        chat_stdin(executable, checkpoint)
        user_commands(executable, checkpoint, fixtures / "bert", checkpoint / "empty cache")
        multi_model_benchmark(executable, checkpoint, fixtures / "bert")
        encoder_batch_checks(executable, fixtures / "bert")
        rerank_checks(executable, fixtures / "bert", checkpoint / "reranker")
        checkpoint_defaults(executable, checkpoint)
