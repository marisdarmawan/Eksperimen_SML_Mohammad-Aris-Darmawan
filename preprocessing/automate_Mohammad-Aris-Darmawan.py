import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
import os
import kagglehub

def preprocess_data(output_path):
    print("Mendownload dataset dari Kaggle via API...")
    # Mendownload dataset dari Kaggle
    dataset_path = kagglehub.dataset_download("ranafayezz/hotel-bookings")
    
    # Menargetkan file spesifik di dalam folder yang diunduh
    raw_data_path = os.path.join(dataset_path, "hotel_bookings_cleaned.csv")
    print(f"Data mentah ditemukan di: {raw_data_path}")
    
    df = pd.read_csv(raw_data_path)
    
    print("Memulai proses data preprocessing...")
    
    # 1. Menangani Missing Values
    for col in df.columns:
        if df[col].isnull().sum() > 0:
            if df[col].dtype in ['int64', 'float64']:
                df[col].fillna(df[col].median(), inplace=True)
            else:
                df[col].fillna(df[col].mode()[0], inplace=True)
                
    # 2. Encoding Data Kategorikal
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    le = LabelEncoder()
    for col in categorical_cols:
        df[col] = le.fit_transform(df[col].astype(str))
        
    # 3. Memisahkan fitur dan target ('is_canceled') untuk proses scaling
    target_col = 'is_canceled'
    X = df.drop(columns=[target_col])
    y = df[target_col]
    
    # 4. Standarisasi / Scaling Fitur
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled_df = pd.DataFrame(X_scaled, columns=X.columns)
    
    # Menggabungkan kembali fitur yang sudah di-scaling dengan kolom target
    df_preprocessed = pd.concat([X_scaled_df, y.reset_index(drop=True)], axis=1)
    
    # 5. Menyimpan dataset yang sudah diproses
    print(f"Menyimpan data hasil preprocessing ke: {output_path}")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df_preprocessed.to_csv(output_path, index=False)
    print("Otomatisasi preprocessing selesai!")

if __name__ == "__main__":
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    OUTPUT_DATA_PATH = os.path.join(BASE_DIR, "preprocessing", "hotel_bookings_preprocessed.csv")
    
    # Jalankan fungsi
    preprocess_data(OUTPUT_DATA_PATH)