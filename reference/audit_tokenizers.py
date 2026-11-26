"""Check each GGUF tokenizer fingerprint against the pinned HF tokenizer fixtures."""

import argparse
import json
from pathlib import Path
import subprocess

from gguf import GGUFReader


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("models/manifest.json"))
    parser.add_argument("--directory", type=Path, required=True)
    parser.add_argument("--controlled", type=Path, required=True)
    parser.add_argument("--hf-tokenizer", type=Path, required=True)
    parser.add_argument("--fixtures", type=Path, required=True)
    parser.add_argument("--probe", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = json.loads(args.manifest.read_text())
    fixtures = json.loads(args.fixtures.read_text())["examples"]
    hf = json.loads(args.hf_tokenizer.read_text())["model"]
    hf_vocab = [token for token, _ in sorted(hf["vocab"].items(), key=lambda pair: pair[1])]
    hf_merges = [" ".join(merge) if isinstance(merge, list) else merge for merge in hf["merges"]]
    sources = [(args.controlled / entry["filename"], entry)
               for entry in manifest["controlled"]["files"]]
    sources.extend((args.directory / source["repo"] / source["revision"] / entry["filename"], entry)
                   for source in manifest["sources"] for entry in source["files"]
                   if entry["filename"].endswith(".gguf"))
    reports = {}
    for path, entry in sources:
        key = entry["inventory"]["tokenizer_fingerprint"]
        if key in reports:
            continue
        reader = GGUFReader(path)
        vocab = reader.fields["tokenizer.ggml.tokens"].contents()
        merges = reader.fields["tokenizer.ggml.merges"].contents()
        output = subprocess.run(
            [args.probe, "tokenize", path, *(fixture["text"] for fixture in fixtures)],
            capture_output=True, check=True, text=True,
        )
        token_ids = [json.loads(line) for line in output.stdout.splitlines()]
        if len(token_ids) != len(fixtures):
            raise ValueError(f"{path}: tokenizer result count mismatch")
        reports[key] = {
            "source": path.name, "artifact_sha256": entry["sha256"],
            "vocab_matches": vocab == hf_vocab, "merges_match": merges == hf_merges,
            "cases": len(fixtures), "differences": [
                {"text": fixture["text"], "hf_ids": fixture["ids"], "llama_cpp_ids": ids}
                for fixture, ids in zip(fixtures, token_ids, strict=True) if ids != fixture["ids"]
            ],
        }
    args.output.write_text(json.dumps(reports, indent=2, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
