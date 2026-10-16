from __future__ import annotations
import os
import numpy as np
from src import data_prep as dp
from src import evaluate as ev
from src import eda

def main() -> None:
    os.makedirs("report/figures", exist_ok=True)
    os.makedirs("results", exist_ok=True)
    
    # 1. Data yüklənməsi və təmizlənməsi
    df = dp.clean(dp.load_raw())
    
    # EDA qrafiklərinin yaradılması (Əgər eda modulunu yaratmısınızsa)
    # eda.make_all(df)
    
    X, y, feat_names = dp.make_features(df)
    
    # 2. Split
    X_tr, y_tr, X_val, y_val, X_te, y_te = dp.train_val_test_split(X, y)
    
    # İmputasiya
    X_tr, X_val, X_te = dp.impute_median(X_tr, X_val, X_te)
    
    # 3. Task B label (threshold YALNIZ train-dən)[cite: 15, 17]
    y_tr_t, thr = dp.make_tier_label(y_tr)
    y_val_t, _ = dp.make_tier_label(y_val, thr)
    y_te_t, _ = dp.make_tier_label(y_te, thr)
    
    print(f"tier threshold (train median): {thr:,.0f} AZN")
    
    # Standartlaşdırma (SVM və s. üçün)
    X_tr_s, X_val_s, X_te_s = dp.standardize(X_tr, X_val, X_te)
    
    print("Pipeline uğurla tamamlandı!")

if __name__ == "__main__":
    main()