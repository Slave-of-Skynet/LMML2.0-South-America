# Colombia — Looks Good to Me! — Bogotá Sound Check

Localize the inserted review text by comparison against the public source corpus.

## Run

Use a separate Python environment. Run from this country directory.

```bash
python -m pip install -r requirements.txt
mkdir -p outputs
python solve.py --zip /path/to/task_colombia.zip
```

## Input and output

The task ZIP/reviews.json and the public Musical Instruments review corpus used by the script.

Obtain task materials from the organizer task card; preserve their original filenames. Generated results belong in `outputs/` or another directory outside Git.

## Files

- [`requirements.txt`](requirements.txt)
- [`solve.py`](solve.py)

Keep submission values on the organizer platform. No final submission value is supplied in these files.
