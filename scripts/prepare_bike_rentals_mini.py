from __future__ import annotations

import argparse
import csv
import io
import random
import zipfile
from pathlib import Path
from urllib.request import urlopen

UCI_ZIP_URL = "https://archive.ics.uci.edu/static/public/275/bike+sharing+dataset.zip"
DEFAULT_OUTPUT_PATH = Path(__file__).resolve().parents[1] / "data" / "bike_rentals_mini.csv"
SAMPLE_SEED = 8
ROWS_PER_HOUR = 10

SOURCE_COLUMNS = ["hr", "temp", "hum", "windspeed", "workingday", "cnt"]
OUTPUT_COLUMNS = [
    "hour",
    "hour_scaled",
    "temperature",
    "humidity",
    "wind_speed",
    "working_day",
    "rental_count",
    "target_scaled",
]


def download_hour_csv() -> str:
    """Download the UCI Bike Sharing Dataset zip and return hour.csv as text."""
    with urlopen(UCI_ZIP_URL, timeout=30) as response:
        zip_bytes = response.read()

    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as archive:
        return archive.read("hour.csv").decode("utf-8")


def read_source_rows(csv_text: str) -> list[dict[str, str]]:
    """Read UCI hour.csv rows and keep each row's original source order."""
    reader = csv.DictReader(io.StringIO(csv_text))
    missing_columns = sorted(set(SOURCE_COLUMNS) - set(reader.fieldnames or []))
    if missing_columns:
        joined_columns = ", ".join(missing_columns)
        raise ValueError(f"source CSV is missing required columns: {joined_columns}")

    rows: list[dict[str, str]] = []
    for source_index, row in enumerate(reader):
        row["_source_index"] = str(source_index)
        rows.append(row)
    return rows


def select_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Select exactly 10 rows for each hour, then restore source row order."""
    rows_by_hour: dict[int, list[dict[str, str]]] = {hour: [] for hour in range(24)}
    for row in rows:
        rows_by_hour[int(row["hr"])].append(row)

    rng = random.Random(SAMPLE_SEED)
    selected_rows: list[dict[str, str]] = []
    for hour in range(24):
        candidates = rows_by_hour[hour]
        if len(candidates) < ROWS_PER_HOUR:
            raise ValueError(f"hour {hour} has only {len(candidates)} rows")
        selected_rows.extend(rng.sample(candidates, ROWS_PER_HOUR))

    return sorted(selected_rows, key=lambda row: int(row["_source_index"]))


def to_teaching_row(source_row: dict[str, str]) -> dict[str, str]:
    """Convert one UCI source row to the small teaching schema."""
    hour = int(source_row["hr"])
    rental_count = int(source_row["cnt"])

    return {
        "hour": str(hour),
        "hour_scaled": repr(hour / 23),
        "temperature": source_row["temp"],
        "humidity": source_row["hum"],
        "wind_speed": source_row["windspeed"],
        "working_day": source_row["workingday"],
        "rental_count": str(rental_count),
        "target_scaled": repr(rental_count / 1000),
    }


def write_teaching_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    """Write the curated teaching rows using only standard-library csv."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as output_file:
        writer = csv.DictWriter(output_file, fieldnames=OUTPUT_COLUMNS, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(to_teaching_row(row))


def build_dataset(source_hour_csv: Path | None, output_path: Path) -> None:
    if source_hour_csv is None:
        csv_text = download_hour_csv()
    else:
        csv_text = source_hour_csv.read_text(encoding="utf-8")

    source_rows = read_source_rows(csv_text)
    selected_rows = select_rows(source_rows)
    write_teaching_csv(selected_rows, output_path)

    print(f"Wrote {len(selected_rows)} rows to {output_path}")
    print(f"Sampling rule: seed={SAMPLE_SEED}, {ROWS_PER_HOUR} rows per hour, sorted by source order")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare the mini bike-rental teaching dataset.")
    parser.add_argument(
        "--source-hour-csv",
        type=Path,
        default=None,
        help="Optional local UCI hour.csv path. If omitted, the script downloads the UCI zip.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT_PATH,
        help="Output path for the curated mini CSV.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    build_dataset(source_hour_csv=args.source_hour_csv, output_path=args.output)


if __name__ == "__main__":
    main()
