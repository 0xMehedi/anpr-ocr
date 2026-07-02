"""
Main augmentation engine.

Reads the training CSV, copies original images,
creates augmented images, and generates labels.csv
and augmentation_report.csv.
"""

from pathlib import Path
import shutil
import random
import cv2
import pandas as pd
from tqdm import tqdm

from augmentation.config import (
    TRAIN_IMAGE_DIR,
    TRAIN_CSV,
    OUTPUT_IMAGE_DIR,
    OUTPUT_CSV,
    REPORT_CSV,
    RANDOM_SEED,
)

from augmentation.pipelines import get_all_pipelines

# ==========================================================
# Initialization
# ==========================================================

random.seed(RANDOM_SEED)

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_CSV.parent.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load Data
# ==========================================================

df = pd.read_csv(TRAIN_CSV)

required_columns = {"filename", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {required_columns}"
    )

pipelines = get_all_pipelines()

labels = []
report = []

# ==========================================================
# Processing
# ==========================================================

print("=" * 60)
print("Generating Augmented Dataset")
print("=" * 60)

for _, row in tqdm(df.iterrows(), total=len(df)):

    filename = row["filename"]
    label = row["label"]

    src = TRAIN_IMAGE_DIR / filename

    image = cv2.imread(str(src))

    if image is None:
        print(f"Cannot read {filename}")
        continue

    # ------------------------------------------------------
    # Copy Original
    # ------------------------------------------------------

    shutil.copy2(
        src,
        OUTPUT_IMAGE_DIR / filename
    )

    labels.append({
        "filename": filename,
        "label": label
    })

    report.append({
        "filename": filename,
        "augmentation": "original"
    })

    # ------------------------------------------------------
    # Generate Augmented Images
    # ------------------------------------------------------

    stem = Path(filename).stem
    suffix = Path(filename).suffix

    for pipeline_name, pipeline in pipelines.items():

        augmented = pipeline(image.copy())

        new_filename = f"{stem}_{pipeline_name}{suffix}"

        cv2.imwrite(
            str(OUTPUT_IMAGE_DIR / new_filename),
            augmented
        )

        labels.append({
            "filename": new_filename,
            "label": label
        })

        report.append({
            "filename": new_filename,
            "augmentation": pipeline_name
        })

# ==========================================================
# Save CSV Files
# ==========================================================

labels_df = pd.DataFrame(labels)
labels_df.to_csv(
    OUTPUT_CSV,
    index=False,
    encoding="utf-8-sig"
)

report_df = pd.DataFrame(report)
report_df.to_csv(
    REPORT_CSV,
    index=False,
    encoding="utf-8-sig"
)

# ==========================================================
# Summary
# ==========================================================

print("\n" + "=" * 60)
print("Augmentation Completed Successfully")
print("=" * 60)

print(f"Original Images : {len(df)}")
print(f"Generated Images: {len(report_df)}")
print(f"Labels Saved    : {OUTPUT_CSV}")
print(f"Report Saved    : {REPORT_CSV}")