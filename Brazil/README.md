# Brazil — Don't Pull My Triggers

Optimize the supplied classifier under a per-tensor relative L2 bound while preserving clean accuracy and a targeted trigger behavior.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python prepare.py
python trigger_search.py
python train.py --rank 0 --steps 200 --every 100 --layers all --name reproduce
python evaluate.py outputs/reproduce/latest.pt --package
```

## Input and output

inputs/task_brazil_data.zip and inputs/baseline_solution.ipynb from the official task folder.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

Organizer material: [source 1](https://drive.google.com/drive/folders/1bRLEszxSn8gNnWswj_ZoORIyUVnWEWsl).

## Files

- [`common.py`](common.py)
- [`evaluate.py`](evaluate.py)
- [`prepare.py`](prepare.py)
- [`requirements.txt`](requirements.txt)
- [`train.py`](train.py)
- [`training_config.json`](training_config.json)
- [`trigger_search.py`](trigger_search.py)
- [`validation.json`](validation.json)

Keep submission values on the organizer platform. No final submission value is supplied in these files.

## Submitted model

The [model.pt release asset](https://github.com/Slave-of-Skynet/LMML2.0-South-America/releases/download/jury-models-20261115/model.pt)
is retained separately from source. Its SHA-256 and measured checks are in `validation.json`.
Validate it with `python evaluate.py /path/to/model.pt` after preparing the official inputs.
The official dataset ZIP SHA-256 is `7fc9dcb07c49134e8fdd7a35533e280854e9dd38b62995e362823623be7db764`.
Use only the trusted organizer reference checkpoint: the supplied architecture uses checkpoint
deserialization and its hash is checked by `common.py` before loading.
