"""Exercise the GGUF oracle's distinct output-head path with a local synthetic variant."""

import argparse
import json
from pathlib import Path

from gguf import GGUFReader, GGUFWriter
import numpy as np
import torch

from generate_fixtures import IDS, load_hf
from gguf_oracle import reconstruct


@torch.inference_mode()
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--controlled-f32", type=Path, required=True)
    parser.add_argument("--output-file", type=Path, required=True)
    args = parser.parse_args()
    if args.output_file.exists():
        parser.error("--output-file already exists")

    reader = GGUFReader(args.controlled_f32)
    writer = GGUFWriter(args.output_file, "llama")
    for name, field in reader.fields.items():
        if not name.startswith("GGUF.") and name != "general.architecture":
            writer.add_key_value(name, field.contents(), *field.types)
    for tensor in reader.tensors:
        writer.add_tensor(tensor.name, tensor.data)
        if tensor.name == "token_embd.weight":
            writer.add_tensor("output.weight", np.negative(tensor.data))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    native = load_hf(args.model)
    model, _ = reconstruct(args.output_file, args.model)
    if model.lm_head.weight is model.model.embed_tokens.weight:
        raise ValueError("output must remain distinct")
    if not torch.equal(model.lm_head.weight, -native.model.embed_tokens.weight):
        raise ValueError("output weights differ from synthetic construction")
    ids = torch.tensor([IDS])
    baseline = -native(ids, use_cache=False).logits
    if not torch.equal(model(ids, use_cache=False).logits, baseline):
        raise ValueError("logits are not the exact negatives of the tied reference")
    print(json.dumps({"output_tied": False, "weights": "exact_negated_embedding", "logits": "exact_negated_native"}))


if __name__ == "__main__":
    main()
