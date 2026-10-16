from src.data_prep import load_raw, clean

def main():
    print("1. Xam data yüklənir...")
    df = load_raw("data/bina_az_sale.csv")
    print(f"Xam datanın ölçüsü: {df.shape}")
    
    print("2. Data təmizlənir...")
    df_clean = clean(df)
    print(f"Təmizlənmiş datanın ölçüsü: {df_clean.shape}")
    
    print("Pipeline testi uğurla tamamlandı!")

if __name__ == "__main__":
    main()