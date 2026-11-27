"""Compare PEFT adapter inference with explicitly merged reference checkpoints."""

import json
import math
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

from serve import chat_fixture, request, server


def read_tensors(path):
    with path.open("rb") as file:
        header = json.loads(file.read(struct.unpack("<Q", file.read(8))[0]))
        data = file.read()
    tensors = {}
    for name, entry in header.items():
        if name != "__metadata__":
            start, end = entry["data_offsets"]
            tensors[name] = (entry["dtype"], entry["shape"], data[start:end])
    return tensors


def write_tensors(path, tensors):
    header, data = {}, bytearray()
    for name, (dtype, shape, values) in tensors.items():
        header[name] = dict(dtype=dtype, shape=shape, data_offsets=[len(data), len(data) + len(values)])
        data.extend(values)
    encoded = json.dumps(header, separators=(",", ":")).encode()
    encoded += b" " * (-len(encoded) % 8)
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + data)


def weights(shape, values, fp16=False):
    return ("F16" if fp16 else "F32", shape,
            struct.pack(f"<{len(values)}{'e' if fp16 else 'f'}", *values))


def add_update(adapter, reference, module, rank, scale, *, embedding=False, transposed=False, bias=False):
    name = module + ".weight"
    dtype, shape, raw = reference[name]
    assert dtype == "F32"
    inputs, outputs = shape if embedding or transposed else shape[::-1]
    a = [((i * 3 + 5) % 11 - 5) / 16 for i in range(rank * inputs)]
    b = [((i * 7 + 3) % 13 - 6) / 16 for i in range(outputs * rank)]
    peft = "base_model.model." + module
    adapter[peft + (".lora_embedding_A" if embedding else ".lora_A.weight")] = weights([rank, inputs], a, True)
    adapter[peft + (".lora_embedding_B" if embedding else ".lora_B.weight")] = weights([outputs, rank], b)
    merged = list(struct.unpack(f"<{inputs * outputs}f", raw))
    for out in range(outputs):
        for index in range(inputs):
            delta = sum(b[out * rank + k] * a[k * inputs + index] for k in range(rank))
            offset = index * outputs + out if embedding or transposed else out * inputs + index
            merged[offset] += scale * delta
    reference[name] = weights(shape, merged)
    if bias:
        name = module + ".bias"
        offset = [(i % 7 - 3) / 8 for i in range(outputs)]
        adapter[peft + ".lora_B.bias"] = weights([outputs], offset)
        values = struct.unpack(f"<{outputs}f", reference[name][2])
        reference[name] = weights([outputs], [value + scale * delta for value, delta in zip(values, offset)])


def fixture(checkpoint, paths, *, gpt2=False):
    path, merged = paths
    for directory in paths:
        directory.mkdir()
    config = json.loads((checkpoint / "config.json").read_text())
    reference = read_tensors(checkpoint / "model.safetensors")
    adapter = {}
    settings = dict(peft_type="LORA", r=2, lora_alpha=3, bias="none", use_bdlora=False)
    if gpt2:
        settings.update(lora_bias=True, fan_in_fan_out=True)
        add_update(adapter, reference, "transformer.h.0.attn.c_attn", 2, 1.5, transposed=True, bias=True)
    else:
        config["tie_word_embeddings"] = False
        reference["lm_head.weight"] = reference["model.embed_tokens.weight"]
        settings.update(use_rslora=True, rank_pattern={"v_proj": 3},
                        alpha_pattern={"^model.layers.0.self_attn.v_proj": 5})
        add_update(adapter, reference, "model.layers.0.self_attn.v_proj", 3, 5 / math.sqrt(3))
        add_update(adapter, reference, "model.layers.1.mlp.down_proj", 2, 3 / math.sqrt(2))
        add_update(adapter, reference, "model.embed_tokens", 2, 3 / math.sqrt(2), embedding=True)
        add_update(adapter, reference, "lm_head", 2, 3 / math.sqrt(2))
    (path / "adapter_config.json").write_text(json.dumps(settings))
    write_tensors(path / "adapter_model.safetensors", adapter)
    (merged / "config.json").write_text(json.dumps(config))
    write_tensors(merged / "model.safetensors", reference)


def call(command, *arguments):
    return json.loads(subprocess.check_output([executable, command, *arguments], timeout=10))


def close(actual, expected):
    assert len(actual) == len(expected)
    assert all(abs(a - b) < 0.00002 for a, b in zip(actual, expected)), (actual, expected)


def check_parity(checkpoint, adapter, merged):
    tokens = "1,2,3,4,5,6,7,8,9"
    expected = call("forward", merged, "--tokens", tokens, "--json")["logits"]
    plain = call("forward", checkpoint, "--tokens", tokens, "--json")["logits"]
    assert max(abs(a - b) for a, b in zip(plain, expected)) > 0.01
    for mode in ([], ["--cached", "--prefill-chunk", "4"]):
        actual = call("forward", checkpoint, "--tokens", tokens, "--json", "--lora", adapter, *mode)["logits"]
        close(actual, expected)


def check_batch(checkpoint, adapter, merged):
    tokens = [1, 3, 4, 5, 6] * 5
    args = ["--tokens", ",".join(map(str, tokens)), "--max-new-tokens", "3", "--greedy", "--json"]
    base_ids = call("generate", checkpoint, *args)["generated_ids"]
    tuned_ids = call("generate", merged, *args)["generated_ids"]
    assert base_ids != tuned_ids
    selection = dict(tokens=tokens, max_new_tokens=3, greedy=True)
    items = [dict(selection, id=1), dict(selection, id=2, lora="tuned"), {"id": 3, "unload_lora": "tuned"}]
    events = subprocess.check_output([executable, "batch", checkpoint, "--prefill-chunk", "4",
                                      "--lora", f"tuned={adapter}", "--lora-root", adapter.parent],
                                     input="".join(json.dumps(item) + "\n" for item in items),
                                     text=True, timeout=10)
    events = [json.loads(event) for event in events.splitlines()]
    for identifier, expected in ((1, base_ids), (2, tuned_ids)):
        assert [event["token_id"] for event in events
                if event["id"] == identifier and event["event"] == "token"] == expected
    assert any(event["id"] == 3 and event.get("code") == "resource_limit" for event in events)


def check_server(checkpoint, adapter, merged, root):
    load, unload = "/v1/load_lora_adapter", "/v1/unload_lora_adapter"
    config = dict(lora_name="tuned", lora_path=adapter.name)
    logits = dict(tokens=[1, 3, 4], output="logits")
    expected = call("infer", merged, "--tokens", "1,3,4", "--output", "logits")["values"]
    terminal = call("run", checkpoint, "b", "--lora", adapter, "--output", "logits")["values"]
    close(terminal, call("infer", merged, "--tokens", "4", "--output", "logits")["values"])
    prefix = dict(tokens=[1, 3, 4, 5, 6] * 5, max_new_tokens=2, greedy=True)
    options = ("--model-id", "base", "--lora", f"tuned={adapter}")
    (root / "escape").symlink_to(Path(__file__).resolve().parent, target_is_directory=True)
    with server(executable, checkpoint, *options) as port:
        request(port, unload, dict(lora_name="tuned"), 403)
    with server(executable, checkpoint, *options, "--lora-root", root) as port:
        assert [card["id"] for card in request(port, "/v1/models")["data"]] == ["base", "tuned"]
        assert request(port, "/v1/models/tuned")["parent"] == "base"
        assert request(port, "/model?model=tuned")["base_model"] == "base"
        close(request(port, "/infer", dict(logits, model="tuned"))["values"], expected)
        assert request(port, "/v1/completions", dict(model="tuned", prompt="b", max_tokens=1))["model"] == "tuned"
        request(port, "/infer", dict(logits, lora="tuned"), 400)
        assert request(port, "/infer", prefix).get("cached_prompt_tokens", 0) == 0
        assert request(port, "/infer", prefix)["cached_prompt_tokens"] >= 16
        assert request(port, "/infer", dict(prefix, model="tuned")).get("cached_prompt_tokens", 0) == 0
        assert request(port, "/infer", dict(prefix, model="tuned"))["cached_prompt_tokens"] >= 16
        request(port, unload, dict(lora_name="tuned"))
        request(port, "/infer", dict(logits, model="tuned"), 404)
        request(port, load, dict(config, lora_name="base"), 400)
        request(port, load, dict(config, lora_path="../"), 400)
        request(port, load, dict(config, lora_path="escape"), 400)
        request(port, load, config)
        assert request(port, "/infer", dict(prefix, model="tuned")).get("cached_prompt_tokens", 0) == 0


def check_invalid(checkpoint, adapter, invalid):
    invalid.mkdir()
    config = json.loads((adapter / "adapter_config.json").read_text())
    config_path, weights_path = invalid / "adapter_config.json", invalid / "adapter_model.safetensors"
    tensors = read_tensors(adapter / weights_path.name)
    a = next(name for name in tensors if name.endswith(".lora_A.weight"))
    b = a.replace(".lora_A.", ".lora_B.")

    def rejected(settings, weights, expected):
        config_path.write_text(json.dumps(settings))
        write_tensors(weights_path, weights)
        result = subprocess.run([executable, "forward", checkpoint, "--tokens", "1,2", "--lora", invalid],
                                capture_output=True, text=True, timeout=10)
        assert result.returncode and expected in result.stderr, result.stderr

    rejected(dict(config, use_dora=True), tensors, "expected a PEFT LoRA")
    rejected(dict(config, use_bdlora=True), tensors, "unsupported LoRA field use_bdlora")
    rejected(config, {name: tensor for name, tensor in tensors.items() if name != b}, "missing")
    rejected(config, dict(tensors, unexpected=tensors[a]), "unsupported LoRA tensor")
    dtype, shape, _ = tensors[a]
    rejected(config, dict(tensors, **{a: weights(shape, [math.nan] * math.prod(shape), dtype == "F16")}),
             "nonfinite LoRA weight")


if __name__ == "__main__":
    executable, fixtures = map(Path, sys.argv[1:])
    with tempfile.TemporaryDirectory(prefix="nathr-lora-") as directory:
        root = Path(directory)
        checkpoint = root / "llama"
        checkpoint.mkdir()
        chat_fixture(checkpoint, fixtures / "model/f32")
        adapter, merged = root / "adapter", root / "merged"
        fixture(checkpoint, (adapter, merged))
        check_parity(checkpoint, adapter, merged)
        check_batch(checkpoint, adapter, merged)
        check_server(checkpoint, adapter, merged, root)
        check_invalid(checkpoint, adapter, root / "invalid")
        gpt2 = fixtures / "gpt2"
        fixture(gpt2, (root / "gpt2-adapter", root / "gpt2-merged"), gpt2=True)
        check_parity(gpt2, root / "gpt2-adapter", root / "gpt2-merged")
