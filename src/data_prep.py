import pandas as pd
import numpy as np

DATA_PATH = "data/bina_az_sale.csv"
LEAKAGE_COLUMNS = ['unit_price', 'total_price', 'owner_name', 'shop_name', 'address']

def load_raw(path=DATA_PATH):
    """
    CSV faylını UTF-8 kodlaşdırması ilə oxuyur və ilkin məlumatı qaytarır.
    """
    df = pd.read_csv(path, encoding='utf-8')
    print(f"Data uğurla yükləndi. Sətir sayı: {df.shape[0]}, Sütun sayı: {df.shape[1]}")
    print("\nSütunlar və boşluq faizləri:")
    print(df.isnull().mean() * 100)
    return df

def clean(df):
    """
    Datanı təmizləyir, sütun adlarını standartlaşdırır və leakage sütunlarını silir.
    """
    df = df.copy()
    
    # 1. Dublikatları silmək
    df.drop_duplicates(inplace=True)
    
    # 2. Leakage və lazımsız sütunları silmək
    cols_to_drop = [col for col in LEAKAGE_COLUMNS if col in df.columns]
    df.drop(columns=cols_to_drop, inplace=True)
    
    # 3. Sahə (area) sütununu təmizləmək
    if 'Sahə' in df.columns:
        df['area'] = df['Sahə'].astype(str).str.replace(r'[^\d.]', '', regex=True)
        df['area'] = pd.to_numeric(df['area'], errors='coerce')
    
    # 4. Qiymət (price) sütununu təmizləmək
    if 'price' in df.columns:
        df['price'] = pd.to_numeric(df['price'], errors='coerce')
        
    # 5. Çatışmayan dəyərlərin doldurulması
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
    
    categorical_cols = df.select_dtypes(include=['object']).columns
    df[categorical_cols] = df[categorical_cols].fillna('unknown')
    
    return df

def make_features(df):
    """
    Kateqorik sütunları encode edir, yüksək unikal dəyəri olan mətn sütunlarını 
    yaddaş problemi yaratmasın deyə çıxarır və X, y massivlərini qaytarır.
    """
    df = df.copy()
    
    if 'price' not in df.columns:
        raise ValueError("Datasetdə 'price' sütunu tapılmadı!")
        
    y = df['price'].values
    X_df = df.drop(columns=['price'])
    
    # Yaddaş problemi yaratmamaq üçün çoxlu unikal dəyəri olan və ya mətn/ID olan sütunları atırıq
    for col in X_df.select_dtypes(include=['object']).columns:
        if X_df[col].nunique() > 50 or col in ['description', 'attributes', 'location', 'address', 'img_url', 'rel_url', 'estate_rel_url_x', 'estate_rel_url_y', 'estate_id', 'id_x', 'id_y', 'datetime_scrape_x', 'datetime_scrape_y']:
            X_df = X_df.drop(columns=[col])
    
    # Qalan kateqorik sütunları one-hot encoding etmək
    X_df = pd.get_dummies(X_df, drop_first=True)
    
    feature_names = X_df.columns.tolist()
    X = X_df.values.astype(float)
    
    return X, y, feature_names

def train_val_test_split(X, y, seed=42):
    """
    Məlumatı 70% train, 15% validation və 15% test olaraq bölür.
    """
    np.random.seed(seed)
    n = len(X)
    indices = np.random.permutation(n)
    
    train_idx = int(n * 0.70)
    val_idx = int(n * 0.85)
    
    tr_indices = indices[:train_idx]
    val_indices = indices[train_idx:val_idx]
    te_indices = indices[val_idx:]
    
    X_tr, y_tr = X[tr_indices], y[tr_indices]
    X_val, y_val = X[val_indices], y[val_indices]
    X_te, y_te = X[te_indices], y[te_indices]
    
    return X_tr, y_tr, X_val, y_val, X_te, y_te

def make_tier_label(y, threshold=None):
    """
    Task B üçün: Qiymət medianından yuxarı olanlar premium (1), qalanlar standart (0) olur.
    """
    if threshold is None:
        threshold = np.median(y)
    
    labels = (y > threshold).astype(int)
    return labels, threshold

def standardize(X_tr, X_val, X_te):
    """
    SVM üçün yalnız train setinin statistikalarından istifadə edərək standartlaşdırma edir.
    """
    mean = np.mean(X_tr, axis=0)
    std = np.std(X_tr, axis=0)
    std[std == 0] = 1.0
    
    X_tr_s = (X_tr - mean) / std
    X_val_s = (X_val - mean) / std
    X_te_s = (X_te - mean) / std
    
    return X_tr_s, X_val_s, X_te_s
if __name__ == "__main__":
    df_raw = load_raw()
    df_cleaned = clean(df_raw)
    print("Təmizlənmiş datanın ölçüsü:", df_cleaned.shape)
    
    X, y, feature_names = make_features(df_cleaned)
    print("X massivinin ölçüsü:", X.shape)
    print("y massivinin ölçüsü:", y.shape)