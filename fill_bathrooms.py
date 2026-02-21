"""
========================================================
  fill_missing_fields.py
  - bathrooms   : fill only where missing
  - land_perches: overwrite ALL rows (existing values wrong)
  - floor_sqft  : skip (already correct in CSV)

  Run:  python fill_missing_fields.py

  Input/Output: data/houses_raw.csv (updated in place)
========================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.lankapropertyweb.com/",
}

CSV_PATH = "data/houses_raw_srilanka_new.csv"


def get_detail_fields(url: str, session: requests.Session) -> dict:
    """
    Fetch detail page and extract from .overview-item blocks:
      Bathrooms/WCs  -> bathrooms
      Area of land   -> land_perches
    """
    result = {}
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        for item in soup.select("div.overview-item"):
            label_el = item.select_one("[class*='label']")
            value_el = item.select_one("[class*='value']")
            if not label_el or not value_el:
                continue

            label = label_el.get_text(strip=True).lower()
            value = value_el.get_text(strip=True)
            nums  = re.findall(r"\d+\.?\d*", value)
            if not nums:
                continue

            if "bathroom" in label:
                result["bathrooms"] = int(float(nums[0]))

            elif "area of land" in label:
                result["land_perches"] = float(nums[0])

    except Exception as e:
        print(f"    Warning: {e}")

    return result


def main():
    df = pd.read_csv(CSV_PATH)
    print(f"Loaded {len(df)} rows")

    if "bathrooms" not in df.columns:
        df["bathrooms"] = None
    if "land_perches" not in df.columns:
        df["land_perches"] = None

    total = len(df)
    print(f"Missing bathrooms  : {df['bathrooms'].isna().sum()}")
    print(f"Land perches       : will be overwritten for ALL {total} rows")
    print(f"Rows to visit      : {total}\n")

    session  = requests.Session()
    filled_b = 0
    filled_l = 0
    failed   = 0

    for i, (idx, row) in enumerate(df.iterrows()):
        url = str(row.get("url", ""))
        if not url.startswith("http"):
            failed += 1
            continue

        print(f"  [{i+1}/{total}] ad={row.get('ad_id','')} | ...{url[-50:]}")
        fields = get_detail_fields(url, session)

        # Bathrooms - only fill if currently missing
        if "bathrooms" in fields and pd.isna(df.at[idx, "bathrooms"]):
            df.at[idx, "bathrooms"] = fields["bathrooms"]
            filled_b += 1

        # Land perches - always overwrite (old values are wrong)
        if "land_perches" in fields:
            df.at[idx, "land_perches"] = fields["land_perches"]
            filled_l += 1

        if fields:
            print(f"    OK {fields}")
        else:
            failed += 1
            print(f"    NOT FOUND")

        # Checkpoint every 50 rows
        if (i + 1) % 50 == 0:
            df.to_csv(CSV_PATH, index=False)
            print(f"    Checkpoint saved (bath={filled_b}, land={filled_l})")

        time.sleep(1)

    # Final save
    df.to_csv(CSV_PATH, index=False)

    print(f"\n{'='*55}")
    print(f"  Done!")
    print(f"  Bathrooms filled   : {filled_b}")
    print(f"  Land perches fixed : {filled_l}")
    print(f"  Could not fetch    : {failed}")
    print(f"  Still missing bath : {df['bathrooms'].isna().sum()}")
    print(f"  Still missing land : {df['land_perches'].isna().sum()}")
    print(f"  Saved -> {CSV_PATH}")
    print(f"{'='*55}")


if __name__ == "__main__":
    main()
