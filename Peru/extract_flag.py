"""Recover an affine-encoded message from ASCII-unreachable embeddings.

Python 3.11+ standard library only; never executes model code or pickle.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import struct


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate JSON key: {key}")
        result[key] = value
    return result


def read_container(path):
    raw = Path(path).read_bytes()
    if len(raw) < 8:
        raise ValueError("Missing safetensors header length")
    header_length = struct.unpack_from("<Q", raw)[0]
    if not 0 < header_length <= min(16*1024*1024, len(raw)-8):
        raise ValueError("Invalid safetensors header length")
    data_start = 8+header_length
    header = json.loads(raw[8:data_start], object_pairs_hook=unique_object)
    if not isinstance(header, dict):
        raise ValueError("Header must be a JSON object")
    intervals = []
    for name, descriptor in header.items():
        if name == "__metadata__":
            if not isinstance(descriptor, dict) or not all(
                    isinstance(k, str) and isinstance(v, str) for k, v in descriptor.items()):
                raise ValueError("Invalid metadata")
            continue
        if not isinstance(descriptor, dict) or descriptor.get("dtype") != "F32":
            raise ValueError(f"Expected F32 tensor: {name}")
        shape, offsets = descriptor.get("shape"), descriptor.get("data_offsets")
        if not isinstance(shape, list) or not all(type(n) is int and n >= 0 for n in shape):
            raise ValueError(f"Invalid shape: {name}")
        if not isinstance(offsets, list) or len(offsets) != 2 or not all(type(n) is int for n in offsets):
            raise ValueError(f"Invalid offsets: {name}")
        start, end = offsets
        if not 0 <= start <= end <= len(raw)-data_start or end-start != math.prod(shape)*4:
            raise ValueError(f"Invalid tensor extent: {name}")
        intervals.append((start, end))
    cursor = 0
    for start, end in sorted(intervals):
        if start != cursor:
            raise ValueError("Tensor data have gaps or overlaps")
        cursor = end
    if cursor != len(raw)-data_start:
        raise ValueError("Unclaimed trailing bytes")
    return raw, header, data_start


def extract(path):
    raw, header, data_start = read_container(path)
    tensor_name = "embedding.weight"
    descriptor = header.get(tensor_name)
    if descriptor is None or descriptor["shape"] != [256, 32]:
        raise ValueError("Expected a 256 x 32 byte embedding")
    width = descriptor["shape"][1]
    tensor_start = data_start+descriptor["data_offsets"][0]
    matches = []
    run = bytearray()
    run_start = None

    def finish():
        nonlocal run, run_start
        if re.fullmatch(rb"SIGMOID_[A-Z]+", run):
            matches.append((run_start, bytes(run)))
        run, run_start = bytearray(), None

    # Strict ASCII has byte IDs 0..127. Scan only the unreachable half.
    for flat_index in range(128*width, 256*width):
        value = struct.unpack_from("<f", raw, tensor_start+flat_index*4)[0]
        code = 128+1024*value
        if math.isfinite(code) and code.is_integer() and 32 <= code <= 126:
            if run_start is None:
                run_start = flat_index
            run.append(int(code))
        else:
            finish()
    finish()
    if len(matches) != 1:
        raise ValueError(f"Expected one encoded flag, found {len(matches)}")
    first, decoded = matches[0]
    start = tensor_start+first*4
    return {
        "flag": decoded.decode("ascii"),
        "model_sha256": hashlib.sha256(raw).hexdigest(),
        "tensor": tensor_name,
        "shape": descriptor["shape"],
        "start_row": first//width,
        "start_column": first%width,
        "float_count": len(decoded),
        "file_offset_start": start,
        "file_offset_end_exclusive": start+4*len(decoded),
        "encoding": "ASCII byte = 128 + 1024 * F32 value",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", nargs="?", type=Path, default=Path("model.safetensors"))
    parser.add_argument("--json", action="store_true", help="Include coordinates and hash")
    args = parser.parse_args()
    result = extract(args.model)
    print(json.dumps(result, indent=2) if args.json else result["flag"])


if __name__ == "__main__":
    main()
