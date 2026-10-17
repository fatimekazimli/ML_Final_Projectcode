from __future__ import annotations

import numpy as np
import pandas as pd

SEED = 42
DATA_PATH = "data/bina_az_sale.csv"
USD_TO_AZN = 1.70

# Real sütun adları (Kaggle "Binaaz sale project" v1, house_sale.csv)
TARGET = "price"
CURRENCY = "currency_x"
LISTING_ID = "estate_rel_url_x"
SCRAPE_TIME = "datetime_scrape_x"
SOURCE = {
    "area": "Sahə",
    "rooms": "Otaq sayı",
    "floor_raw": "Mərtəbə",
    "land_raw": "Torpaq sahəsi",
    "lat": "lat",
    "lng": "lng",
    "building_type": "Binanın növü",
    "category": "Kateqoriya",
    "repair": "repair",
    "mortgage": "mortgage",
    "deed": "bill_of_sale",
    "city": "city",
    "location": "location",
}

NUMERIC = ["area", "rooms", "floor", "total_floors", "land_area", "lat", "lng"]
CATEGORICAL = ["building_type", "category", "repair", "mortgage", "deed", "city", "location"]

PRICE_MIN, PRICE_MAX = 5_000, 20_000_000
AREA_MIN, AREA_MAX = 10, 2_000
PRICE_BANDS = np.array([50_000, 100_000, 200_000, 350_000, 500_000, 1_000_000])
TOP_LOCATIONS = 25
VAL_SIZE, TEST_SIZE = 0.15, 0.15


def to_number(s):
    s = (s.astype(str)
         .str.replace(" ", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False))
    return pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path, encoding="utf-8", low_memory=False)
    print(f"raw shape: {df.shape}")
    return df


def clean(raw):
    n_raw = len(raw)

    # 1) Eyni elanın təkrar scrape-ləri: URL üzrə yalnız ən son versiya saxlanılır
    # stabil sıralama: hər kompüterdə və pandas versiyasında eyni sətirlər saxlanılır
    raw = raw.sort_values(SCRAPE_TIME, kind="mergesort").drop_duplicates(LISTING_ID, keep="last")
    n_unique = len(raw)

    # 2) Hədəf: AZN-ə çevrilir (USD varsa)
    price = pd.to_numeric(raw[TARGET], errors="coerce")
    usd = raw[CURRENCY].astype(str).str.upper().eq("USD")
    price = price.where(~usd, price * USD_TO_AZN)

    # 3) Mətn kimi gələn rəqəmlər: "85 m²", "5 / 12", "6 sot"
    floor = raw[SOURCE["floor_raw"]].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
    land = to_number(raw[SOURCE["land_raw"]])
    land = land.where(~raw[SOURCE["land_raw"]].astype(str).str.contains("sot", case=False), land * 100)

    # 4) Yalnız açıq siyahıdakı sütunlar saxlanılır -> leakage və ID-lər avtomatik düşür
    data = pd.DataFrame({
        "price": price,
        "area": to_number(raw[SOURCE["area"]]),
        "rooms": to_number(raw[SOURCE["rooms"]]),
        "floor": pd.to_numeric(floor[0], errors="coerce"),
        "total_floors": pd.to_numeric(floor[1], errors="coerce"),
        "land_area": land,
        "lat": pd.to_numeric(raw[SOURCE["lat"]], errors="coerce"),
        "lng": pd.to_numeric(raw[SOURCE["lng"]], errors="coerce"),
        "building_type": raw[SOURCE["building_type"]],
        "category": raw[SOURCE["category"]],
        "repair": raw[SOURCE["repair"]],
        "mortgage": raw[SOURCE["mortgage"]],
        "deed": raw[SOURCE["deed"]],
        "city": raw[SOURCE["city"]],
        "location": raw[SOURCE["location"]],
    })

    # 5) Sabit, əl ilə seçilmiş sərhədlər (datadan hesablanmır -> leakage yoxdur)
    keep = data["price"].between(PRICE_MIN, PRICE_MAX)
    keep &= data["area"].isna() | data["area"].between(AREA_MIN, AREA_MAX)
    data = data.loc[keep].reset_index(drop=True)

    used = set(SOURCE.values()) | {TARGET}
    excluded = [c for c in raw.columns if c not in used]
    leakage = [c for c in excluded if "price" in c.lower()]
    info = {
        "raw_rows": n_raw,
        "unique_listing_rows": n_unique,
        "clean_rows": len(data),
        "usd_rows": int(usd.sum()),
        "missing_or_out_of_range_price": int((~price.between(PRICE_MIN, PRICE_MAX)).sum()),
        "excluded_columns": excluded,
        "leakage_columns": leakage,
    }
    print(f"rows: raw={n_raw}  unique listings={n_unique}  clean={len(data)}  usd rows={info['usd_rows']}")
    print("leakage columns dropped:", leakage)
    print("other excluded columns:", [c for c in excluded if c not in leakage])
    return data, info


def split_indices(y, seed=SEED):
    rng = np.random.default_rng(seed)
    strata = np.digitize(y, PRICE_BANDS)
    train, validation, test = [], [], []
    for stratum in np.unique(strata):
        idx = rng.permutation(np.flatnonzero(strata == stratum))
        n_test = round(len(idx) * TEST_SIZE)
        n_val = round(len(idx) * VAL_SIZE)
        test.extend(idx[:n_test])
        validation.extend(idx[n_test:n_test + n_val])
        train.extend(idx[n_test + n_val:])
    train, validation, test = np.asarray(train), np.asarray(validation), np.asarray(test)
    print(f"split sizes: train={len(train)}  val={len(validation)}  test={len(test)}")
    return train, validation, test


def encode(data, train, validation, test):
    # medianlar, top rayonlar və one-hot sütunları YALNIZ train-dən öyrənilir
    train_frame = data.iloc[train]
    medians = train_frame[NUMERIC].median().fillna(0)
    counts = train_frame["location"].fillna("unknown").value_counts().sort_index()
    counts = counts.sort_values(ascending=False, kind="mergesort")
    common_locations = set(counts.head(TOP_LOCATIONS).index)
    frames = []
    for idx in (train, validation, test):
        frame = data.iloc[idx]
        numeric = frame[NUMERIC].fillna(medians).reset_index(drop=True).astype(float)
        cats = frame[CATEGORICAL].fillna("unknown").astype(str).reset_index(drop=True)
        cats["location"] = cats["location"].where(cats["location"].isin(common_locations), "other")
        frames.append(pd.concat([numeric, pd.get_dummies(cats, dtype=float)], axis=1))
    names = frames[0].columns.tolist()
    arrays = tuple(f.reindex(columns=names, fill_value=0).to_numpy(dtype=float) for f in frames)
    print(f"features: {len(names)}  (numeric={len(NUMERIC)}, one-hot={len(names) - len(NUMERIC)})")
    return arrays, names


def make_tier_label(y_price, threshold=None):
    if threshold is None:
        threshold = float(np.median(y_price))
    return (np.asarray(y_price) > threshold).astype(int), threshold


def standardize(X_tr, *others):
    mean = X_tr.mean(axis=0)
    std = X_tr.std(axis=0)
    std = np.where(std == 0, 1.0, std)
    return [(A - mean) / std for A in (X_tr, *others)]


def prepare(path=DATA_PATH):
    data, info = clean(load_raw(path))
    y = data["price"].to_numpy(dtype=float)
    train, val, test = split_indices(y)
    (X_tr, X_val, X_te), names = encode(data, train, val, test)
    y_tr, y_val, y_te = y[train], y[val], y[test]
    t_tr, thr = make_tier_label(y_tr)
    t_val, _ = make_tier_label(y_val, thr)
    t_te, _ = make_tier_label(y_te, thr)
    print(f"tier threshold (train median): {thr:,.0f} AZN   premium share train/val/test: "
          f"{t_tr.mean():.3f} / {t_val.mean():.3f} / {t_te.mean():.3f}")
    info.update({"train_rows": len(train), "validation_rows": len(val), "test_rows": len(test),
                 "feature_count": len(names), "tier_threshold_AZN": thr,
                 "class_balance": {"train": int(t_tr.sum()), "validation": int(t_val.sum()), "test": int(t_te.sum())}})
    return {
        "data": data, "info": info, "names": names,
        "train": train, "val": val, "test": test,
        "X_tr": X_tr, "X_val": X_val, "X_te": X_te,
        "y_tr": y_tr, "y_val": y_val, "y_te": y_te,
        "t_tr": t_tr, "t_val": t_val, "t_te": t_te, "thr": thr,
    }


if __name__ == "__main__":
    d = prepare()
    print(d["names"])
