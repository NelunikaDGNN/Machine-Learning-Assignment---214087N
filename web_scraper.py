"""
========================================================
  FILE 1: 1_scraper.py
  STEP 1 — Scrape ALL Sri Lanka house listings
           from lankapropertyweb.com

  Run:  pip install requests beautifulsoup4 pandas
        python 1_scraper.py

  Output: data/houses_raw.csv  (~4000 rows)
========================================================
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import time
import re
import os

# ── Create data folder ───────────────────────────────────────────────────────
os.makedirs("data", exist_ok=True)

# ── Settings ─────────────────────────────────────────────────────────────────
BASE_URL       = "https://www.lankapropertyweb.com/sale/index.php"
PROPERTY_TYPE  = "House"
TOTAL_PAGES    = 134       # ~4010 listings / 30 per page
DELAY_SECONDS  = 2         # polite delay

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.lankapropertyweb.com/",
}


# ── Price parser ─────────────────────────────────────────────────────────────
def parse_price(text: str):
    """Convert 'Rs. 45M', 'Rs. 45,000,000' → 45000000.0"""
    if not text:
        return None
    t = text.upper().replace(",", "").replace("RS.", "").replace("RS", "").strip()
    # strip anything after first non-numeric/non-M character group
    t = t.split("NEGOTIABLE")[0].split("PER")[0].strip()
    try:
        if "M" in t:
            num = re.sub(r"[^\d.]", "", t.split("M")[0])
            return float(num) * 1_000_000 if num else None
        elif "MILLION" in t:
            num = re.sub(r"[^\d.]", "", t.replace("MILLION", ""))
            return float(num) * 1_000_000 if num else None
        elif "LAKH" in t or "LAK" in t:
            num = re.sub(r"[^\d.]", "", re.sub(r"LAKHS?", "", t))
            return float(num) * 100_000 if num else None
        else:
            num = re.sub(r"[^\d.]", "", t)
            return float(num) if num else None
    except ValueError:
        return None


# ── Parse one listing card ────────────────────────────────────────────────────
def parse_card(card) -> dict:
    item = {}

    # ── Ad ID ──────────────────────────────────────────────────────────────
    item["ad_id"] = card.get("data-ad-id", "")

    # ── Suburb / Location ──────────────────────────────────────────────────
    loc_el = card.select_one("span.location")
    item["suburb"] = loc_el.get_text(strip=True) if loc_el else ""

    # ── Title ──────────────────────────────────────────────────────────────
    title_el = card.select_one("h4.listing-title a")
    item["title"] = title_el.get_text(strip=True) if title_el else ""

    # ── Full Address ───────────────────────────────────────────────────────
    addr_el = card.select_one("h5.listing-address")
    if addr_el:
        # remove the icon text
        addr_text = addr_el.get_text(strip=True)
        item["address"] = addr_text
    else:
        item["address"] = ""

    # ── Price ──────────────────────────────────────────────────────────────
    price_el = card.select_one("div.listing-price")
    raw_price = ""
    if price_el:
        # remove child spans (like "Negotiable") and get price text only
        raw_price = price_el.get_text(strip=True)
    item["price_raw"] = raw_price
    item["price_lkr"] = parse_price(raw_price)

    # ── Beds & Floor area from listing-summery ─────────────────────────────
    # Structure: li[img alt="bed icon"] > span.count   → bedrooms
    #            li[img alt="floor area icon"] > span.count + span.unit  → sqft
    item["bedrooms"]   = None
    item["floor_sqft"] = None

    for li in card.select("div.listing-summery ul li"):
        imgs = li.select("img")
        counts = li.select("span.count")
        units  = li.select("span.unit")
        if not counts:
            continue
        val_text = counts[0].get_text(strip=True)
        unit_text = units[0].get_text(strip=True).lower() if units else ""
        alt_text  = imgs[0].get("alt", "").lower() if imgs else ""

        if "bed" in alt_text:
            try:
                item["bedrooms"] = int(val_text)
            except ValueError:
                pass
        elif "floor" in alt_text or "scale" in alt_text or "sqft" in unit_text:
            try:
                item["floor_sqft"] = float(val_text)
            except ValueError:
                pass

    # ── URL ────────────────────────────────────────────────────────────────
    a_tag = card.select_one("a.listing-header")
    href = a_tag.get("href", "") if a_tag else ""
    item["url"] = ("https://www.lankapropertyweb.com" + href) if href.startswith("/") else href

    return item


# ── Scrape detail page for land & bathrooms ────────────────────────────────
def scrape_detail(url: str, session: requests.Session) -> dict:
    """Visit individual listing page to get bathrooms and land size."""
    extra = {}
    if not url:
        return extra
    try:
        resp = session.get(url, headers=HEADERS, timeout=15)
        soup = BeautifulSoup(resp.text, "html.parser")

        # Try structured table/list rows on detail page
        for row in soup.select("table tr, .property-details li, .features li, dl dt, .detail-row"):
            text = row.get_text(" ", strip=True).lower()
            nums = re.findall(r"\d+\.?\d*", text)
            if not nums:
                continue
            if "bathroom" in text and "bathrooms" not in extra:
                extra["bathrooms"] = int(float(nums[0]))
            if "perch" in text and "land_perches" not in extra:
                extra["land_perches"] = float(nums[0])
            if ("sq" in text or "floor" in text) and "floor_sqft" not in extra:
                extra["floor_sqft"] = float(nums[0])

        # Fallback: search anywhere in page text
        if "bathrooms" not in extra:
            bath_el = soup.find(string=re.compile(r"bathroom", re.I))
            if bath_el:
                nums = re.findall(r"\d+", bath_el.parent.get_text() if bath_el.parent else "")
                if nums:
                    extra["bathrooms"] = int(nums[0])

        if "land_perches" not in extra:
            land_el = soup.find(string=re.compile(r"perch", re.I))
            if land_el:
                nums = re.findall(r"\d+\.?\d*", land_el.parent.get_text() if land_el.parent else "")
                if nums:
                    extra["land_perches"] = float(nums[0])

    except Exception as e:
        pass
    return extra


# ── Main ──────────────────────────────────────────────────────────────────────
def run_scraper():
    all_listings = []
    session = requests.Session()

    print(f"\n{'='*60}")
    print(f"  Scraping ALL Sri Lanka Houses — {TOTAL_PAGES} pages")
    print(f"{'='*60}\n")

    for page in range(1, TOTAL_PAGES + 1):
        # Page 1 uses a different URL format
        if page == 1:
            url = f"https://www.lankapropertyweb.com/forsale-all-{PROPERTY_TYPE}.html"
        else:
            url = f"{BASE_URL}?page={page}&property-type={PROPERTY_TYPE}"

        print(f"  Page {page:>3}/{TOTAL_PAGES} | {url}")

        try:
            resp = session.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            soup = BeautifulSoup(resp.text, "html.parser")

            # ── Find all listing cards ──────────────────────────────────────
            cards = soup.select("article.listing-item")

            if not cards:
                print(f"    ⚠  No cards found on page {page} — stopping early")
                break

            page_listings = []
            for card in cards:
                item = parse_card(card)
                if item.get("price_lkr"):   # skip if no price
                    page_listings.append(item)

            # ── Fetch detail pages for bathrooms & land ────────────────────
            for item in page_listings:
                needs_detail = (
                    item.get("bathrooms") is None or
                    item.get("land_perches") is None
                )
                if needs_detail and item.get("url"):
                    extra = scrape_detail(item["url"], session)
                    for k, v in extra.items():
                        if item.get(k) is None:
                            item[k] = v
                    time.sleep(0.5)

            all_listings.extend(page_listings)
            print(f"    ✅ {len(page_listings)} listings | Total so far: {len(all_listings)}")

            # ── Save checkpoint every 10 pages ─────────────────────────────
            if page % 10 == 0:
                df_temp = pd.DataFrame(all_listings)
                df_temp.to_csv("data/houses_raw_srilanka_new.csv", index=False, encoding="utf-8")
                print(f"    💾 Checkpoint saved ({len(df_temp)} rows)")

        except requests.RequestException as e:
            print(f"    ❌ Request failed on page {page}: {e}")

        time.sleep(DELAY_SECONDS)

    # ── Final save ────────────────────────────────────────────────────────────
    df = pd.DataFrame(all_listings)

    # Remove duplicates by ad_id
    df.drop_duplicates(subset=["ad_id"], keep="first", inplace=True)
    df.reset_index(drop=True, inplace=True)

    out_path = "data/houses_raw_srilanka_new.csv"
    df.to_csv(out_path, index=False, encoding="utf-8")

    print(f"\n{'='*60}")
    print(f"  ✅ DONE! {len(df)} unique listings saved → {out_path}")
    print(f"  Columns: {list(df.columns)}")
    print(f"\n  Preview:")
    print(df[["title", "price_lkr", "bedrooms", "floor_sqft", "suburb"]].head(10).to_string())
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_scraper()
