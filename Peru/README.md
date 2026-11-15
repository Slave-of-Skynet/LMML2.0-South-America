# Peru — The Knots You Cannot Read

Parse the safetensors container and recover the affine-encoded message from ASCII-unreachable embedding rows.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python extract_flag.py /path/to/model.safetensors
```

## Input and output

The official model.safetensors. It is read as bytes; the extractor does not execute checkpoint code.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1jSvIFGcLw11QUQxHPFlLLfZQohDeszkb).

## Files

- [`extract_flag.py`](extract_flag.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
