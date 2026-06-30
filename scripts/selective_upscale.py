from pathlib import Path
import shutil
import cv2
import pandas as pd
from tqdm import tqdm

# ==========================
# Configuration
# ==========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "cropped_1676"
OUTPUT_DIR = PROJECT_ROOT / "data" / "upscaled"
MODEL_PATH = PROJECT_ROOT / "models" / "FSRCNN_x2.pb"
ANALYSIS_CSV = PROJECT_ROOT / "results" / "resolution_analysis.csv"
REPORT_CSV = PROJECT_ROOT / "results" / "upscale_report.csv"

UPSCALE_PERCENTILE = 25
MODEL_NAME = "fsrcnn"
SCALE = 2

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ==========================
# Load Super Resolution Model
# ==========================

sr = cv2.dnn_superres.DnnSuperResImpl_create()
sr.readModel(str(MODEL_PATH))
sr.setModel(MODEL_NAME, SCALE)

# ==========================
# Read Resolution Analysis
# ==========================

df = pd.read_csv(ANALYSIS_CSV)

threshold = df["area"].quantile(UPSCALE_PERCENTILE / 100)

print("=" * 60)
print(f"Area Threshold ({UPSCALE_PERCENTILE}th percentile): {threshold:.2f}")
print("=" * 60)

report = []

# ==========================
# Process Images
# ==========================

for _, row in tqdm(df.iterrows(), total=len(df)):

    filename = row["filename"]
    area = row["area"]

    src = IMAGE_DIR / filename
    dst = OUTPUT_DIR / filename

    if area < threshold:

        image = cv2.imread(str(src))

        if image is None:
            print(f"Failed to read {filename}")
            continue

        upscaled = sr.upsample(image)

        cv2.imwrite(str(dst), upscaled)

        status = "Yes"

    else:

        shutil.copy2(src, dst)

        status = "No"

    report.append({
        "filename": filename,
        "area": area,
        "threshold": threshold,
        "upscaled": status
    })

# ==========================
# Save Report
# ==========================

report_df = pd.DataFrame(report)
report_df.to_csv(REPORT_CSV, index=False)

print("\nDone!")
print(f"Images Processed : {len(report_df)}")
print(f"Report Saved     : {REPORT_CSV}")