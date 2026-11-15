# Ecuador — Ciclopaseo Mirage

Optimize a targeted image perturbation within the permitted per-channel RGB bound.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python main.py --input /path/to/test-input.png --output outputs/test-output.png
```

## Input and output

The input image and classifier/mobilenet_v3_large-5c1a4163.pth from the official task bundle.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`challenge_model.py`](challenge_model.py)
- [`main.py`](main.py)
- [`requirements.txt`](requirements.txt)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
