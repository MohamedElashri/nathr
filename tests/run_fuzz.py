"""Run short parser and quantization fuzz tests with the checked-in fixtures."""

import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
from tempfile import TemporaryDirectory


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("build", type=Path)
    args = parser.parse_args()
    build = args.build.resolve()
    fixtures = Path(__file__).parent / "fixtures"
    seeds = {
        "safetensors": {"model": (fixtures / "model/f32/model.safetensors").read_bytes()},
        "gguf": {"model": (fixtures / "tiny.gguf").read_bytes()},
        "tokenizer": {"model": (fixtures / "model/tokenizer.json").read_bytes(),
                      "utf8": "A\n中文\u200d🙂مرحبا".encode(), "partial": b"a\xf0\x9f"},
        "quant": {},
    }
    for path in sorted((fixtures / "quant").glob("*.json")):
        for name, fixture in json.loads(path.read_text()).items():
            for i, block in enumerate(fixture["blocks"]):
                seeds["quant"][f"{name}-{i}"] = bytes([fixture["type_id"]]) + bytes.fromhex(block)

    def run(name):
        with TemporaryDirectory(dir=build, prefix=f"fuzz-{name}-") as directory:
            corpus = Path(directory)
            for filename, data in seeds[name].items():
                (corpus / filename).write_bytes(data)
            command = [build / f"nathr_fuzz_{name}", "-seed=58053", "-max_total_time=10",
                       "-timeout=5", f"-max_len={1024 if name == 'quant' else 65536}",
                       f"-artifact_prefix={build}/{name}-", corpus]
            result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
            if result.returncode:
                raise SystemExit(result.stdout)
            print(f"{name} passed", flush=True)

    with ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(run, seeds))


if __name__ == "__main__":
    main()
