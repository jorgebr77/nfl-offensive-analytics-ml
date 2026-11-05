
from pathlib import Path
from urllib.request import urlretrieve

DATA_DIR = Path(__file__).resolve().parent / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = (
    "https://github.com/nflverse/nflverse-data/"
    "releases/download/pbp"
)

for year in range(2018, 2025):
    filename = f"play_by_play_{year}.parquet"
    destination = DATA_DIR / filename

    if destination.exists() and destination.stat().st_size > 0:
        print(f"Already exists: {filename}")
        continue

    url = f"{BASE_URL}/{filename}"
    print(f"Downloading {filename}...")

    try:
        urlretrieve(url, destination)
        print(f"Saved: {destination}")
    except Exception:
        destination.unlink(missing_ok=True)
        raise

print("NFL data download complete.")
