from __future__ import annotations
import numpy as np
import pandas as pd

SEED = 42
DATA_PATH = "data/bina_az_sale.csv"
USD_TO_AZN = 1.70

# Real sütun adları (Addım 0 yoxlamasına uyğun olaraq tənzimlənib)
LEAKAGE_COLUMNS = ["unit_price", "total_price"]
ID_COLUMNS = [
    "id_x", "id_y", "estate_id", "rel_url", "img_url", 
    "estate_rel_url_x", "estate_rel_url_y", "datetime_scrape_x", 
    "datetime_scrape_y", "owner_name", "shop_name", "address", 
    "description", "attributes"
]

RENAME = {
    "Saha": "area", 
    "Otaq sayı": "rooms", 
    "Mərtəbə": "floor_raw",
    "Torpaq sahəsi": "land_area", 
    "Binanın növü": "building_type",
    "Kateqoriya": "category", 
    "Təmir": "repair", 
    "İpoteka": "mortgage",
    "Çıxarış": "deed"
}

AREA_MIN, AREA_MAX = 10, 2000  # m2
PRICE_MIN, PRICE_MAX = 5_000, 20_000_000  # AZN
MIN_LOCATION_COUNT = 30

NUM_COLS = ["area", "rooms", "floor", "total_floors", "land_area", "lat", "lng"]
CAT_COLS = ["building_type", "category", "repair", "mortgage", "deed", "city"]


def to_number(s):
    s = (s.astype(str)
         .str.replace("\u00a0", "", regex=False)
         .str.replace(" ", "", regex=False)
         .str.replace(",", ".", regex=False))
    return pd.to_numeric(s.str.extract(r"(\d+(?:\.\d+)?)")[0], errors="coerce")


def load_raw(path=DATA_PATH):
    df = pd.read_csv(path, encoding="utf-8")
    print(f"raw shape: {df.shape}")
    print((df.isna().mean() * 100).round(1).to_string())
    return df


def clean(df):
    df = df.copy()
    n0 = len(df)
    
    # 1) Leakage və ID sütunlarının əvvəlcədən silinməsi (B3)
    drop_cols = [c for c in LEAKAGE_COLUMNS + ID_COLUMNS if c in df.columns]
    df = df.drop(columns=drop_cols)
    print("dropped columns:", drop_cols)
    
    # 2) Sütun adlarının tərcüməsi
    df = df.rename(columns={k: v for k, v in RENAME.items() if k in df.columns})
    
    # 3) Dublikatların silinməsi (ID-lər silindikdən sonra düzgün işləyir)
    df = df.drop_duplicates()
    print(f"duplicates removed: {n0 - len(df)}")
    
    # 4) Hədəf (price): boş və ya <= 0 qiymətlər silinir (B2: fillna yoxdur)
    if "price" in df.columns:
        df["price"] = to_number(df["price"])
        if "currency" in df.columns:
            usd = df["currency"].astype(str).str.contains(r"USD|\$", case=False)
            df.loc[usd, "price"] *= USD_TO_AZN
        df = df[df["price"].notna() & (df["price"] > 0)]
    
    # 5) Rəqəmsal sütunların parse edilməsi (B4)
    for c in ["area", "rooms", "land_area", "lat", "lng"]:
        if c in df.columns:
            df[c] = to_number(df[c])
            
    if "floor_raw" in df.columns:
        fl = df["floor_raw"].astype(str).str.extract(r"(\d+)\s*/\s*(\d+)")
        df["floor"] = pd.to_numeric(fl[0], errors="coerce")
        df["total_floors"] = pd.to_numeric(fl[1], errors="coerce")
        df = df.drop(columns=["floor_raw"])
        
    # 6) Outlier-lərin təmizlənməsi (sabit sərhədlər)
    n1 = len(df)
    if "price" in df.columns:
        df = df[df["price"].between(PRICE_MIN, PRICE_MAX)]
    if "area" in df.columns:
        df = df[df["area"].isna() | df["area"].between(AREA_MIN, AREA_MAX)]
    print(f"outliers removed: {n1 - len(df)}, final shape: {df.shape}")
    
    return df.reset_index(drop=True)


def make_features(df):
    y = df["price"].to_numpy(dtype=float)
    X_num = df[[c for c in NUM_COLS if c in df.columns]].astype(float)
    cats = df[[c for c in CAT_COLS if c in df.columns]].fillna("unknown").astype(str)
    
    if "location" in df.columns:
        loc = df["location"].fillna("unknown").astype(str)
        counts = loc.value_counts()
        cats["location"] = loc.where(loc.map(counts) > MIN_LOCATION_COUNT, "other")
        
    X_cat = pd.get_dummies(cats, dtype=float)
    X_df = pd.concat([X_num, X_cat], axis=1)
    print(f"X: {X_df.shape} (numeric={X_num.shape[1]}, one-hot={X_cat.shape[1]})")
    return X_df.to_numpy(dtype=float), y, X_df.columns.tolist()


def train_val_test_split(X, y, val_size=0.15, test_size=0.15, seed=SEED, n_bins=10):
    rng = np.random.default_rng(seed)
    edges = np.quantile(y, np.linspace(0, 1, n_bins + 1)[1:-1])
    strata = np.digitize(y, edges)
    
    tr, va, te = [], [], []
    for s in np.unique(strata):
        idx = rng.permutation(np.where(strata == s)[0])
        n_te = int(round(len(idx) * test_size))
        n_va = int(round(len(idx) * val_size))
        te.extend(idx[:n_te])
        va.extend(idx[n_te:n_te + n_va])
        tr.extend(idx[n_te + n_va:])
        
    tr, va, te = np.array(tr), np.array(va), np.array(te)
    print(f"split sizes train={len(tr)} val={len(va)} test={len(te)}")
    return X[tr], y[tr], X[va], y[va], X[te], y[te]


def impute_median(X_tr, *others):
    med = np.nanmedian(X_tr, axis=0)
    med = np.where(np.isnan(med), 0.0, med)
    out = []
    for A in (X_tr, *others):
        A = A.copy()
        r, c = np.where(np.isnan(A))
        A[r, c] = med[c]
        out.append(A)
    return out


def make_tier_label(y_price, threshold=None):
    if threshold is None:
        threshold = float(np.median(y_price))
    return (np.asarray(y_price) > threshold).astype(int), threshold


def standardize(X_tr, X_val, X_te):
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std = np.where(std == 0, 1.0, std)
    return (X_tr - mean) / std, (X_val - mean) / std, (X_te - mean) / std