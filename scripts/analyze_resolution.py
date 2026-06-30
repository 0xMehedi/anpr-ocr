from pathlib import Path
import cv2
import pandas as pd
import matplotlib.pyplot as plt


# ==========================
# Configuration
# ==========================
PROJECT_ROOT = Path(__file__).resolve().parent.parent

IMAGE_DIR = PROJECT_ROOT / "data" / "cropped_1676"
LABELS_CSV = PROJECT_ROOT / "data" / "labels.csv"
RESULTS_DIR = PROJECT_ROOT / "results"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# ==========================
# Load CSV
# ==========================
def load_labels():
    if not LABELS_CSV.exists():
        raise FileNotFoundError(f"CSV not found: {LABELS_CSV}")

    df = pd.read_csv(LABELS_CSV)

    if "filename" not in df.columns:
        raise ValueError("CSV must contain a 'filename' column.")

    return df


# ==========================
# Analyze Images
# ==========================
def analyze_images(df):
    records = []
    missing_images = []

    for filename in df["filename"]:

        image_path = IMAGE_DIR / filename

        if not image_path.exists():
            missing_images.append(filename)
            continue

        image = cv2.imread(str(image_path))

        if image is None:
            missing_images.append(filename)
            continue

        height, width = image.shape[:2]

        area = width * height
        aspect_ratio = round(width / height, 3)
        longest_side = max(width, height)
        shortest_side = min(width, height)

        records.append({
            "filename": filename,
            "width": width,
            "height": height,
            "area": area,
            "aspect_ratio": aspect_ratio,
            "longest_side": longest_side,
            "shortest_side": shortest_side
        })

    result_df = pd.DataFrame(records)

    return result_df, missing_images


# ==========================
# Save CSV
# ==========================
def save_analysis(df):

    output_csv = RESULTS_DIR / "resolution_analysis.csv"
    df.to_csv(output_csv, index=False)

    print(f"\nResolution analysis saved to:")
    print(output_csv)
#
#percentile analysys

def generate_percentiles(df):

    percentile_file = RESULTS_DIR / "resolution_percentiles.txt"

    percentiles = [5, 10, 25, 50, 75, 90, 95]

    with open(percentile_file, "w", encoding="utf-8") as f:

        f.write("========== Resolution Percentiles ==========\n\n")

        for column in ["width", "height", "area"]:

            f.write(f"----- {column.upper()} -----\n")

            values = df[column].quantile([p / 100 for p in percentiles])

            for p, value in zip(percentiles, values):
                f.write(f"{p:>2}% : {value:.2f}\n")

            f.write("\n")

    print("Percentile report saved to:")
    print(percentile_file)





# ==========================
# Summary Statistics
# ==========================
def generate_summary(df, missing_images):

    summary_file = RESULTS_DIR / "resolution_summary.txt"

    with open(summary_file, "w", encoding="utf-8") as f:

        f.write("========== Resolution Analysis ==========\n\n")

        f.write(f"Total Images in CSV     : {len(df) + len(missing_images)}\n")
        f.write(f"Successfully Processed  : {len(df)}\n")
        f.write(f"Missing Images          : {len(missing_images)}\n\n")

        for column in ["width", "height", "area"]:

            f.write(f"----- {column.upper()} -----\n")
            f.write(f"Minimum : {df[column].min()}\n")
            f.write(f"Maximum : {df[column].max()}\n")
            f.write(f"Mean    : {df[column].mean():.2f}\n")
            f.write(f"Median  : {df[column].median():.2f}\n")
            f.write(f"Std Dev : {df[column].std():.2f}\n\n")

        if missing_images:
            f.write("Missing Images:\n")
            for img in missing_images:
                f.write(f"{img}\n")

    print("Summary saved to:")
    print(summary_file)


# ==========================
# Histogram
# ==========================
def plot_histogram(df, column):

    plt.figure(figsize=(8, 5))

    plt.hist(df[column], bins=30)

    plt.title(f"{column.capitalize()} Distribution")
    plt.xlabel(column.capitalize())
    plt.ylabel("Frequency")

    plt.tight_layout()

    output = RESULTS_DIR / f"{column}_distribution.png"

    plt.savefig(output, dpi=300)
    plt.close()


# ==========================
# Boxplot
# ==========================
def plot_boxplot(df, column):

    plt.figure(figsize=(6, 4))

    plt.boxplot(df[column], vert=True)

    plt.title(f"{column.capitalize()} Boxplot")

    plt.tight_layout()

    output = RESULTS_DIR / f"{column}_boxplot.png"

    plt.savefig(output, dpi=300)
    plt.close()


# ==========================
# Small Images
# ==========================
def export_smallest_images(df):

    smallest = df.sort_values("area").head(100)

    output = RESULTS_DIR / "smallest_images.csv"

    smallest.to_csv(output, index=False)

    print("Smallest image list saved to:")
    print(output)



#percentile








# ==========================
# Main
# ==========================
def main():

    print("=" * 50)
    print("Bangladesh ANPR OCR")
    print("Resolution Analysis")
    print("=" * 50)

    labels = load_labels()

    analysis_df, missing = analyze_images(labels)

    save_analysis(analysis_df)

    generate_summary(analysis_df, missing)

    generate_percentiles(analysis_df)

    for column in ["width", "height", "area"]:
        plot_histogram(analysis_df, column)
        plot_boxplot(analysis_df, column)

    export_smallest_images(analysis_df)

    print("\nAnalysis Complete!")
    print(f"Processed Images : {len(analysis_df)}")
    print(f"Missing Images   : {len(missing)}")


if __name__ == "__main__":
    main()