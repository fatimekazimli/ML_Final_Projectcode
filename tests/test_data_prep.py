import numpy as np
import pandas as pd

from src import data_prep as dp


def fake_raw():
    rows = [
        ("/a", "2026-01-01", 100_000, "AZN", "80 m²", "3", "5 / 9", None, "R1"),
        ("/a", "2026-02-01", 110_000, "AZN", "80 m²", "3", "5 / 9", None, "R1"),
        ("/b", "2026-01-05", 300_000, "AZN", "150 m²", "5", None, "6 sot", "R2"),
        ("/c", "2026-01-07", 50_000, "USD", "45 m²", "1", "2 / 5", None, "R1"),
        ("/d", "2026-01-08", None, "AZN", "60 m²", "2", "3 / 9", None, "R3"),
        ("/e", "2026-01-09", 1, "AZN", "60 m²", "2", "3 / 9", None, "R3"),
        ("/f", "2026-01-10", 200_000, "AZN", "5 m²", "1", "1 / 2", None, "R3"),
    ]
    df = pd.DataFrame(rows, columns=["estate_rel_url_x", "datetime_scrape_x", "price", "currency_x", "Sahə",
                                     "Otaq sayı", "Mərtəbə", "Torpaq sahəsi", "location"])
    for col, val in [("lat", 40.4), ("lng", 49.8), ("Binanın növü", "Yeni tikili"), ("Kateqoriya", "Mənzil"),
                     ("repair", "var"), ("mortgage", None), ("bill_of_sale", "var"), ("city", "Bakı")]:
        df[col] = val
    df["unit_price"] = 1250.0
    df["owner_name"] = "x"
    return df


def test_clean_rules():
    data, info = dp.clean(fake_raw())
    print(data)
    assert info["raw_rows"] == 7 and info["unique_listing_rows"] == 6
    assert len(data) == 3
    assert sorted(data["price"].tolist()) == [85_000, 110_000, 300_000]
    a = data[data["price"] == 110_000].iloc[0]
    assert a["floor"] == 5 and a["total_floors"] == 9 and a["area"] == 80
    assert data[data["price"] == 300_000].iloc[0]["land_area"] == 600
    assert "unit_price" in info["leakage_columns"]
    assert "unit_price" not in data.columns and "owner_name" not in data.columns


def test_encode_uses_train_only():
    data = pd.DataFrame({
        "price": [1.0, 2, 3, 4], "area": [10.0, np.nan, 30, 1000], "rooms": 1.0, "floor": 1.0,
        "total_floors": 1.0, "land_area": np.nan, "lat": 0.0, "lng": 0.0, "building_type": "x",
        "category": "x", "repair": "x", "mortgage": "x", "deed": "x", "city": "x",
        "location": ["A", "A", "B", "Z"],
    })
    (X_tr, X_val, X_te), names = dp.encode(data, np.array([0, 1, 2]), np.array([3]), np.array([3]))
    area = names.index("area")
    assert X_tr[1, area] == 20.0
    assert "location_Z" not in names
    assert X_val[0, names.index("location_A")] == 0 and X_val[0, names.index("location_B")] == 0
    assert not np.isnan(X_tr).any()


def test_tier_threshold_and_standardize_use_train():
    t_tr, thr = dp.make_tier_label(np.array([1, 2, 3, 4, 100]))
    t_te, _ = dp.make_tier_label(np.array([2.5, 3.5]), thr)
    assert thr == 3 and t_te.tolist() == [0, 1]
    A, B = dp.standardize(np.array([[0.0, 5], [2, 5]]), np.array([[4.0, 5]]))
    assert np.allclose(A.mean(axis=0), 0) and B[0, 0] == 3 and B[0, 1] == 0
