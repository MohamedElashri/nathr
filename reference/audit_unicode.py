"""Compare Unicode data categories with the exact tokenizers wheel used as the oracle."""

import argparse
import json
from pathlib import Path

from tokenizers import Regex
from tokenizers.pre_tokenizers import Digits, Split


def category_points(path):
    numbers = set()
    letters = set()
    start = None
    for line in path.read_text().splitlines():
        code, name, category, *_ = line.split(";")
        point = int(code, 16)
        if name.endswith(", First>"):
            start = point
            continue
        if name.endswith(", Last>") and start is None:
            raise ValueError(f"missing start for {name}")
        target = {"L": letters, "N": numbers}.get(category[0])
        if target is not None:
            target.update(range(point if start is None else start, point + 1))
        start = None
    return letters, numbers


def property_points(splitter):
    matched = set()
    for start, end in ((0, 0xD800), (0xE000, 0x110000)):
        for offset in range(start, end, 4096):
            chunk = range(offset, min(offset + 4096, end))
            text = "|" + "|".join(map(chr, chunk)) + "|"
            for _, (begin, finish) in splitter.pre_tokenize_str(text):
                if begin % 2 == 1 and finish == begin + 1:
                    matched.add(chunk[begin // 2])
    return matched


def ranges(points):
    groups = []
    for point in sorted(points):
        if groups and point == groups[-1][1] + 1:
            groups[-1][1] = point
        else:
            groups.append([point, point])
    return [[f"U+{first:04X}", f"U+{last:04X}"] for first, last in groups]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unicode_data", type=Path)
    args = parser.parse_args()
    letters, numbers = category_points(args.unicode_data)
    spaces = {*range(0x9, 0xE), 0x20, 0x85, 0xA0, 0x1680, *range(0x2000, 0x200B),
              0x2028, 0x2029, 0x202F, 0x205F, 0x3000}
    checks = [
        ("digits_is_numeric", Digits(individual_digits=True), numbers),
        ("byte_level_letter", Split(Regex(r"\p{L}"), "isolated"), letters),
        ("byte_level_number", Split(Regex(r"\p{N}"), "isolated"), numbers),
        ("byte_level_whitespace", Split(Regex(r"\s"), "isolated"), spaces),
    ]
    report = {}
    for name, splitter, expected in checks:
        actual = property_points(splitter)
        report[name] = {
            "matches": len(actual),
            "only_unicode_data": ranges(expected - actual),
            "only_tokenizers": ranges(actual - expected),
        }
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
