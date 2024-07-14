import os
import gzip
import json
import jsonlines

from typing import List


def make_printv(verbose: bool):
    def print_v(*args, **kwargs):
        if verbose:
            kwargs["flush"] = True
            print(*args, **kwargs)
        else:
            pass
    return print_v


def read_jsonl(path: str) -> List[dict]:
    if not os.path.exists(path):
        raise FileNotFoundError(f"File `{path}` does not exist.")
    elif not path.endswith(".jsonl"):
        raise ValueError(f"File `{path}` is not a jsonl file.")
    items = []
    with jsonlines.open(path) as reader:
        for item in reader:
            items += [item]
    return items


def write_jsonl(path: str, data: List[dict], append: bool = False):
    with jsonlines.open(path, mode='a' if append else 'w') as writer:
        for item in data:
            writer.write(item)


def read_jsonl_gz(path: str) -> List[dict]:
    if not path.endswith(".jsonl.gz"):
        raise ValueError(f"File `{path}` is not a jsonl.gz file.")
    with gzip.open(path, "rt") as f:
        data = [json.loads(line) for line in f]
    return data
