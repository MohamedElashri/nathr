"""Verify rounded FP16 KV, chunked/parallel execution and cache quality."""

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from gguf import GGMLQuantizationType, GGUFReader, MODEL_ARCH, get_tensor_name_map
import numpy as np
from safetensors.numpy import load_file
import torch
import torch.nn.functional as F

from generate_fixtures import IDS, load_hf
from gguf_oracle import reconstruct
from phase1_verify import capture, command, compare
from phase2_verify import summary
from phase3_verify import corpus_windows, log_probabilities, quality, stream_rows, top5
from rounded_oracle import rounded_operations


def logits_windows(model, windows):
    for window in windows:
        ids = torch.tensor([window["ids"]])
        yield model(ids, use_cache=True).logits[0].numpy()[window["skip"]:]


def probabilities(row):
    logp = log_probabilities(row)
    return logp, np.exp(logp), int(np.argmax(row)), top5(row)


@contextmanager
def integer_operations(model, artifact):
    # Use the independently read serialized signed codes and scales. PyTorch
    # block matmul in FP32 represents each 32-term integer sum exactly.
    reader = GGUFReader(artifact)
    tensors = {tensor.name: tensor for tensor in reader.tensors}
    config = model.config
    name_map = get_tensor_name_map(MODEL_ARCH.LLAMA, config.num_hidden_layers)
    matrices = {}
    for name, parameter in model.state_dict(keep_vars=True).items():
        key = name_map.get_name(name, try_suffixes=(".weight",))
        if name == "lm_head.weight" and config.tie_word_embeddings:
            key = "token_embd.weight"
        tensor = tensors[key]
        if id(parameter) in matrices or tensor.tensor_type != GGMLQuantizationType.Q8_0:
            continue
        rows, columns = parameter.shape
        blocks = np.asarray(tensor.data).reshape(rows, columns // 32, 34)
        if name.endswith(("q_proj.weight", "k_proj.weight")):
            heads = config.num_key_value_heads if name.endswith("k_proj.weight") else config.num_attention_heads
            blocks = blocks.reshape(heads, -1, 2, columns // 32, 34).transpose(0, 2, 1, 3, 4).reshape(blocks.shape)
        codes = np.ascontiguousarray(blocks[..., 2:].view(np.int8).transpose(1, 2, 0), dtype=np.float32)
        scales = blocks[..., :2].copy().view("<f2").reshape(rows, -1)
        matrices[id(parameter)] = (torch.from_numpy(codes), torch.from_numpy(scales.T.astype(np.float64)))

    with rounded_operations(model):
        exact_linear = F.linear

        def integer_linear(x, weight, bias=None):
            if id(weight) not in matrices:
                return exact_linear(x, weight, bias)
            codes, weight_scales = matrices[id(weight)]
            blocks = x.reshape(-1, codes.shape[0], 32).transpose(0, 1)
            scales = blocks.abs().amax(dim=-1, keepdim=True) / 127
            safe_scale = torch.where(scales == 0, 1, scales)
            signed = (blocks.double() / safe_scale.double()).round().clamp(-127, 127).float()
            dots = signed @ codes
            result = (scales.double() * weight_scales[:, None, :] * dots.double()).sum(0).float()
            return result.reshape(*x.shape[:-1], weight.shape[0])

        with patch.object(F, "linear", integer_linear):
            yield


def verify(args, artifact, model, windows, staging):
    with rounded_operations(model):
        fp32 = [[probabilities(row) for row in rows] for rows in logits_windows(model, windows)]

    runtime = {}
    with rounded_operations(model, round_cache=True):
        reference = capture(model, IDS, round_cache=True)
        for backend, threads, chunk in (("scalar", 1, 1), ("auto", args.threads, 3),
                                        ("auto", args.threads, 16)):
            trace = staging / "cache.safetensors"
            command(args.binary, "forward", artifact, "--tokens", ",".join(map(str, IDS)),
                    "--cached", "--kv-cache", "f16", "--backend", backend, "--threads", str(threads),
                    "--prefill-chunk", str(chunk), "--context", "512", "--trace", trace)
            runtime[f"{backend}/{threads}/{chunk}"] = summary(compare(reference, load_file(trace)))

        nonzero = 0

        def checked_rows():
            nonlocal nonzero
            for window, base, expected in zip(windows, fp32, logits_windows(model, windows), strict=True):
                options = ("--kv-cache", "f16", "--prefill-chunk", "7", "--threads", str(args.threads))
                for (row, target, original), ref in zip(stream_rows(args.binary, artifact, window, base, options),
                                                        expected, strict=True):
                    if compare({"logits": ref[None]}, {"logits": row[None]})["logits"]["relative_l2"] > 0:
                        nonzero += 1
                    yield row, target, original

        cache_quality = quality(checked_rows())
    return {"traces": runtime, "cache_quality_fp16_vs_fp32": cache_quality,
            "streamed_rows_with_nonzero_error": nonzero}


def verify_integer(args, artifact, model, windows, staging):
    with rounded_operations(model):
        baseline = [[probabilities(row) for row in rows] for rows in logits_windows(model, windows)]
    runtime = {}
    with integer_operations(model, artifact):
        reference = capture(model, IDS)
        for chunk in (1, 9):
            trace = staging / "integer.safetensors"
            command(args.binary, "forward", artifact, "--tokens", ",".join(map(str, IDS)),
                    "--activation", "q8-int8", "--backend", "avx2", "--threads", str(args.threads),
                    "--cached", "--prefill-chunk", str(chunk), "--context", "512", "--trace", trace)
            runtime[str(chunk)] = summary(compare(reference, load_file(trace)))

        def checked_rows():
            for window, base, expected in zip(windows, baseline, logits_windows(model, windows), strict=True):
                options = ("--activation", "q8-int8", "--prefill-chunk", "7", "--threads", str(args.threads))
                for (row, target, original), ref in zip(stream_rows(args.binary, artifact, window, base, options),
                                                        expected, strict=True):
                    compare({"logits": ref[None]}, {"logits": row[None]})
                    yield row, target, original

        metrics = quality(checked_rows())
    return {"traces": runtime, "quality_int8_vs_fp32": metrics}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("binary", "native", "controlled", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    parser.add_argument("--threads", type=int, default=4, choices=range(1, 65))
    parser.add_argument("--integer", action="store_true", help="verify the Q8_0 activation experiment")
    args = parser.parse_args()
    torch.set_num_threads(1)
    windows, corpus = corpus_windows(Path("fixtures/phase3/corpus.json"), args.native)
    results = {}
    with TemporaryDirectory(prefix="nathr-phase4-") as work:
        for recipe in (("Q8_0",) if args.integer else ("native", "Q8_0", "Q4_0")):
            if recipe == "native":
                artifact, model = args.native, load_hf(args.native)
            else:
                artifact = args.controlled / f"SmolLM2.{recipe}.gguf"
                model, _ = reconstruct(artifact, args.native)
            check = verify_integer if args.integer else verify
            results[recipe] = check(args, artifact, model, windows, Path(work))
            print(f"qualified {recipe} against independent PyTorch", flush=True)
    args.output.write_text(json.dumps({"corpus": corpus, "runtime_threads": args.threads,
                                       "artifacts": results}, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
