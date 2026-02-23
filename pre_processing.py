"""
========================================================
  FILE 2: pre_processing.py
  STEP 2 — Clean and preprocess scraped house data

  Input:  data/houses_raw_srilanka_new.csv
  Output: data/houses_clean_new_one.csv
========================================================
"""

import pandas as pd
import numpy as np
import os

os.makedirs("data", exist_ok=True)

# ── Load ──────────────────────────────────────────────────────────────────────
df = pd.read_csv("data/houses_raw_srilanka_new.csv")
print(f"Loaded: {len(df)} rows")
print(f"Columns: {list(df.columns)}\n")

raw_count = len(df)

# ── 1. Drop duplicates ────────────────────────────────────────────────────────
if "ad_id" in df.columns:
    before = len(df)
    df.drop_duplicates(subset=["ad_id"], keep="first", inplace=True)
    print(f"Removed duplicate ad_ids : {before - len(df)}")

# ── 2. Drop rows with no price ────────────────────────────────────────────────
before = len(df)
df = df[df["price_lkr"].notna() & (df["price_lkr"] > 0)]
print(f"Removed missing price    : {before - len(df)}")

# ── 3. Extract district from suburb + address + title ─────────────────────────
DISTRICT_MAP = {
    "Colombo"      : ["colombo", "dehiwala", "mount lavinia", "kotte", "kaduwela",
                      "maharagama", "moratuwa", "ratmalana", "thalawathugoda",
                      "nugegoda", "boralesgamuwa", "battaramulla", "rajagiriya",
                      "kolonnawa", "homagama", "padukka", "kesbewa", "piliyandala",
                      "angoda", "malabe", "athurugiriya", "hendala", "pelawatta",
                      "nawala", "hokandara", "pannipitiya", "kohuwala", "kottawa",
                      "kalubowila", "rathmalana", "mattegoda", "meegoda", "attidiya",
                      "kahathuduwa", "mirihana", "pepiliyana", "madiwela", "gothatuwa",
                      "koswatta", "godagama", "katubedda", "polgasowita", "delkanda",
                      "thalahena", "nawinna", "madapatha", "kotikawatta", "peliyagoda",
                      "bolgoda", "gonapola", "kirindiwela", "raddolugama","Pelawatte","Pamunuwa",
                      "Bokundara","Kiriwatthuduwa" ,"Habarakada","Udahamulla","Panagoda",
                      "Rattanapitiya","Mulleriyawa","Nawagamuwa","Korathota","Thalapathpitiya"],
    "Gampaha"      : ["gampaha", "negombo", "ja-ela", "ja ela", "wattala",
                      "kandana", "ragama", "kiribathgoda", "kelaniya", "minuwangoda",
                      "divulapitiya", "mirigama", "bandarawatta", "katunayake",
                      "seeduwa", "ekala", "nittambuwa", "veyangoda", "kadawatha",
                      "weliweriya", "kalalgoda", "delgoda", "dankotuwa", "ganemulla",
                      "dompe", "karapitiya", "midigama","Welisara","Makola","Kapuwatta","Heiyantuduwa","Kirillawala",
                      "Kotugoda", "Yakkala","Kadawala"],
    "Kalutara"     : ["kalutara", "panadura", "horana", "beruwala", "aluthgama",
                      "matugama", "bandaragama", "ingiriya", "wadduwa", "payagala",
                      "dodangoda", "bulathsinhala","Moragahahena"],
    "Kandy"        : ["kandy", "peradeniya", "gampola", "nawalapitiya",
                      "katugastota", "kundasale", "digana", "ampitiya",
                      "akurana", "kadugannawa", "teldeniya","Pilimathalawa","Aniwatte","Naranwala"],
    "Galle"        : ["galle", "hikkaduwa", "ambalangoda", "elpitiya",
                      "karandeniya", "baddegama", "bentota", "ahangama",
                      "unawatuna", "koggala", "habaraduwa","Kathaluwa","Kosgoda","Ahungalla","Bope"],
    "Matara"       : ["matara", "weligama", "dikwella", "akuressa", "hakmana",
                      "deniyaya", "kamburupitiya", "mirissa","Madiha"],
    "Kurunegala"   : ["kurunegala", "kuliyapitiya", "nikaweratiya", "pannala",
                      "wariyapola", "mawathagama", "ibbagamuwa", "dambadeniya","Alawwa"],
    "Ratnapura"    : ["ratnapura", "embilipitiya", "balangoda", "kahawatta",
                      "eheliyagoda", "kuruwita", "pelmadulla","Embuldeniya"],
    "Matale"       : ["matale", "dambulla", "galewela", "sigiriya", "rattota"],
    "Nuwara Eliya" : ["nuwara eliya", "hatton", "talawakele", "ginigathena"],
    "Anuradhapura" : ["anuradhapura", "kekirawa", "medawachchiya", "mihintale"],
    "Polonnaruwa"  : ["polonnaruwa", "hingurakgoda", "medirigiriya","Jayanthipura"],
    "Badulla"      : ["badulla", "bandarawela", "haputale", "mahiyanganaya",
                      "ella", "welimada", "passara","Beragala","Jayaminipura"],
    "Hambantota"   : ["hambantota", "tangalle", "beliatta", "tissamaharama",
                      "ambalantota", "weeraketiya" ,"Mattala"],
    "Puttalam"     : ["puttalam", "chilaw", "wennappuwa", "marawila",
                      "anamaduwa", "nattandiya","Madampe"],
    "Trincomalee"  : ["trincomalee", "kinniya", "muttur"],
    "Batticaloa"   : ["batticaloa", "kattankudy", "eravur"],
    "Ampara"       : ["ampara", "kalmunai", "sammanthurai", "akkaraipattu","Dehiattakandiya"],
    "Jaffna"       : ["jaffna", "point pedro", "chavakachcheri", "nallur",
                      "navatkuli", "karainagar"],
    "Vavuniya"     : ["vavuniya", "mankulam"],
    "Kegalle"      : ["kegalle", "mawanella", "warakapola", "rambukkana",
                      "ruwanwella", "galigamuwa","Bulathkohupitiya"],
    "Monaragala"   : ["monaragala", "wellawaya", "bibile", "buttala"],
    "Mullaitivu"   : ["mullaitivu", "oddusuddan"],
    "Kilinochchi"  : ["kilinochchi"],
    "Mannar"       : ["mannar"],
}

def extract_district(row):
    combined = " ".join([
        str(row.get("suburb",  "") or ""),
        str(row.get("address", "") or ""),
        str(row.get("title",   "") or ""),
    ]).lower()
    for district, keywords in DISTRICT_MAP.items():
        for kw in keywords:
            if kw in combined:
                return district
    return "Other"

df["district"] = df.apply(extract_district, axis=1)
print(f"\nDistrict distribution:")
print(df["district"].value_counts().to_string())

# ── 4. Price outlier filter ───────────────────────────────────────────────────
before = len(df)
df = df[(df["price_lkr"] >= 2_000_000) & (df["price_lkr"] <= 600_000_000)]
print(f"\nRemoved price outliers   : {before - len(df)}")

# ── 5. Clean bedrooms ─────────────────────────────────────────────────────────
df["bedrooms"] = pd.to_numeric(df["bedrooms"], errors="coerce")
df.loc[df["bedrooms"] > 10, "bedrooms"] = np.nan
df.loc[df["bedrooms"] < 1,  "bedrooms"] = np.nan

# ── 6. Clean bathrooms ────────────────────────────────────────────────────────
df["bathrooms"] = pd.to_numeric(df["bathrooms"], errors="coerce")
df.loc[df["bathrooms"] > 10, "bathrooms"] = np.nan
df.loc[df["bathrooms"] < 1,  "bathrooms"] = np.nan

# ── 7. Clean land_perches ─────────────────────────────────────────────────────
df["land_perches"] = pd.to_numeric(df["land_perches"], errors="coerce")
df.loc[df["land_perches"] > 400, "land_perches"] = np.nan
df.loc[df["land_perches"] < 0.5, "land_perches"] = np.nan

# ── 8. Clean floor_sqft ───────────────────────────────────────────────────────
df["floor_sqft"] = pd.to_numeric(df["floor_sqft"], errors="coerce")
df.loc[df["floor_sqft"] > 15000, "floor_sqft"] = np.nan
df.loc[df["floor_sqft"] < 100,   "floor_sqft"] = np.nan

# ── 9. Fill missing with district median, fallback to global median ───────────
for col in ["bedrooms", "bathrooms", "land_perches", "floor_sqft"]:
    df[col] = df.groupby("district")[col].transform(
        lambda x: x.fillna(x.median())
    )
    df[col] = df[col].fillna(df[col].median())

# ── 10. Log-transform price (makes distribution normal for ML) ────────────────
df["price_log"] = np.log1p(df["price_lkr"])

# ── 11. Keep only columns needed for ML ───────────────────────────────────────
keep = ["ad_id", "suburb", "district",
        "bedrooms", "bathrooms", "land_perches", "floor_sqft",
        "price_lkr", "price_log"]
keep = [c for c in keep if c in df.columns]
df = df[keep].reset_index(drop=True)

# ── Summary ───────────────────────────────────────────────────────────────────
print(f"\n{'='*50}")
print(f"  Raw rows      : {raw_count}")
print(f"  Clean rows    : {len(df)}")
print(f"  Dropped total : {raw_count - len(df)}")
print(f"\n  Missing values after cleaning:")
print(df.isnull().sum().to_string())
print(f"\n  Feature stats:")
print(df[["bedrooms","bathrooms","land_perches","floor_sqft","price_lkr"]].describe().round(2).to_string())
print(f"{'='*50}")

# ── Save ──────────────────────────────────────────────────────────────────────
df.to_csv("data/houses_clean_new_one.csv", index=False)
print(f"\n  Saved -> data/houses_clean_new_one.csv ({len(df)} rows)")