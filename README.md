# Invoice Automation and Anomaly Detection

A small pipeline that takes invoice images, runs OCR on them, pulls out the important fields, checks them against a few sanity rules, and finally flags anything that looks suspicious using classical machine learning.

## What is inside

- `src/preprocessing.py` cleans up an invoice image with OpenCV.
- `src/ocr_engine.py` runs Tesseract on a cleaned image and returns raw text.
- `src/parser.py` uses regex to turn the raw text into structured fields.
- `src/validator.py` applies rule based checks (math and dates).
- `src/anomaly_detector.py` runs per vendor z-scores, an Isolation Forest, and fuzzy duplicate matching.
- `src/generate_samples.py` builds a folder of synthetic invoice PNGs so the pipeline can be demonstrated end to end.
- `src/run_pipeline.py` is the glue script that produces the final CSV and figures.

## Setup

1. Install Python 3.10 or newer.
2. Install the Tesseract binary. On Windows the easiest source is the UB Mannheim build at https://github.com/UB-Mannheim/tesseract/wiki. During installation note the install path (usually `C:\Program Files\Tesseract-OCR\tesseract.exe`).
3. If Tesseract is not on your PATH, set the environment variable `TESSERACT_CMD` to that path, or edit `src/ocr_engine.py`.
4. From this folder run:

```
pip install -r requirements.txt
python src/generate_samples.py
python src/run_pipeline.py
```

## Outputs

- `output/invoices_extracted.csv` with `is_anomaly` and `anomaly_reason` columns.
- `report/figures/` with the box plot, scatter plot and histogram used in the report.
- `report/report.md` is the written technical report.

## Web app

There is also a small Flask front end in `app.py`. It lets a visitor upload a single invoice image, see the extracted fields plus rule based warnings, and browse the pre-computed sample dataset with anomalous rows highlighted.

Run it locally with:

```
python app.py
```

Then open `http://localhost:5000` in a browser.

## Deploying to Render

The project ships with a `render.yaml` and an `apt.txt`, which together tell Render to install the Tesseract binary during the build and run the Flask app under Gunicorn. To deploy:

1. Push the whole `invoice_pipeline` folder to a GitHub repository.
2. Sign in at https://dashboard.render.com and click **New** then **Web Service**.
3. Connect the GitHub repository. Render will read `render.yaml` and pre-fill the build command, start command and environment variables.
4. Keep the free plan selected and click **Create Web Service**.
5. The first build takes about 3 to 5 minutes while Render installs Tesseract and the Python dependencies. When it finishes, the app is available at a URL like `https://invoice-pipeline-xxxx.onrender.com`.

Notes on the free plan:

- The service sleeps after 15 minutes of inactivity. The first request after a sleep takes around 30 seconds while the container spins back up.
- Uploaded files are processed in memory and discarded, so nothing is persisted between requests.
- The batch anomaly signals (Isolation Forest, cross-invoice duplicate detection) only run inside `run_pipeline.py`. Single uploads through the web app use rule based checks only.
