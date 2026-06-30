from pathlib import Path
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split

# ==========================================================
# Configuration
# ==========================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "upscaled"
LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"

DATA_DIR = PROJECT_ROOT / "data"

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

RANDOM_STATE = 42

# ==========================================================
# Output Directories
# ==========================================================

TRAIN_DIR = DATA_DIR / "train" / "images"
VAL_DIR = DATA_DIR / "val" / "images"
TEST_DIR = DATA_DIR / "test" / "images"

ANNOTATION_DIR = DATA_DIR / "annotations"

for d in [TRAIN_DIR, VAL_DIR, TEST_DIR, ANNOTATION_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ==========================================================
# Load Labels
# ==========================================================

df = pd.read_csv(LABELS_CSV)

required_columns = {"filename", "label"}

if not required_columns.issubset(df.columns):
    raise ValueError(
        f"CSV must contain columns: {required_columns}"
    )

print(f"Total Images : {len(df)}")

# ==========================================================
# Split Dataset
# ==========================================================

train_df, temp_df = train_test_split(
    df,
    train_size=TRAIN_RATIO,
    shuffle=True,
    random_state=RANDOM_STATE
)

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.5,
    shuffle=True,
    random_state=RANDOM_STATE
)

print(f"Train : {len(train_df)}")
print(f"Val   : {len(val_df)}")
print(f"Test  : {len(test_df)}")

# ==========================================================
# Copy Images
# ==========================================================

def export_split(dataframe, image_output_dir, csv_output_path):

    copied = 0
    missing = 0

    for _, row in dataframe.iterrows():

        src = IMAGE_DIR / row["filename"]
        dst = image_output_dir / row["filename"]

        if src.exists():
            shutil.copy2(src, dst)
            copied += 1
        else:
            print(f"Missing image: {row['filename']}")
            missing += 1

    dataframe.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    print(f"\nSaved: {csv_output_path.name}")
    print(f"Copied Images : {copied}")
    print(f"Missing Images: {missing}")


export_split(
    train_df,
    TRAIN_DIR,
    ANNOTATION_DIR / "train.csv"
)

export_split(
    val_df,
    VAL_DIR,
    ANNOTATION_DIR / "val.csv"
)

export_split(
    test_df,
    TEST_DIR,
    ANNOTATION_DIR / "test.csv"
)

print("\n======================================")
print("Dataset split completed successfully!")
print("======================================")