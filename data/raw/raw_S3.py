import pandas as pd
import boto3
import os

# --- CONFIGURAÇÕES ---
file_path = r"C:\Users\Matheus\Downloads\archive\lung_cancer_dataset.csv"
parquet_path = "lung_cancer.parquet"
bucket_name = "dados-raw-lungcancer"
s3_key = "data/lung_cancer.parquet"

df = pd.read_csv(file_path)

df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

s3 = boto3.client('s3')

s3.upload_file(parquet_path, bucket_name, s3_key)
print(f"Arquivo {parquet_path} enviado para s3://{bucket_name}/{s3_key}")

os.remove(parquet_path)
