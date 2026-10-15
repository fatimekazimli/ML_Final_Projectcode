import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge

from data_prep import load_raw, make_features
from evaluate import evaluate_regression

def main():
    print("1. Data yüklənir...")
    df_raw = load_raw()
    
    print("2. Xüsusiyyətlər hazırlanır...")
    X, y, feature_names = make_features(df_raw)
    
    # Boş xanaları (NaN) 0 ilə əvəz edirik ki, model xəta verməsin
    X = np.nan_to_num(X, nan=0.0)
    
    # Həmçinin y-də NaN varsa onları təmizləyirik
    valid_idx = ~np.isnan(y)
    X = X[valid_idx]
    y = y[valid_idx]
    
    print(f"X massivinin ölçüsü: {X.shape}, y massivinin ölçüsü: {y.shape}")
    
    # Datanı Train və Test olaraq bölürük (80% train, 20% test)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    print("3. Model öyrədilir (Ridge Regression)...")
    model = Ridge(alpha=1.0)
    model.fit(X_train, y_train)
    
    print("4. Test dəsti üzrə proqnoz verilir...")
    y_pred = model.predict(X_test)
    
    print("5. Metriklər hesablanır...")
    metrics = evaluate_regression(y_test, y_pred)
    
    print("\n--- MODELİN QİYMƏTLƏNDİRİLMƏ NƏTİCƏLƏRİ ---")
    for metric_name, val in metrics.items():
        print(f"{metric_name}: {val:.4f}")

if __name__ == "__main__":
    main()