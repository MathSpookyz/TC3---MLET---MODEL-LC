import pandas as pd
import boto3
import os

# --- CONFIGURAÇÕES ---
csv_path = r"C:\Users\Matheus\Downloads\archive\lung_cancer_dataset.csv"                # caminho do CSV local
parquet_path = "lung_cancer.parquet"        # caminho do parquet local
bucket_name = "dados-raw-lungcancer"                  # nome do bucket S3
s3_key = "data/lung_cancer.parquet"         # chave/“pasta” no S3

# --- 1. Ler CSV ---
df = pd.read_csv(csv_path)

# --- 2. Converter para Parquet (com compressão snappy ou gzip) ---
df.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

# --- 3. Upload para S3 ---
# credenciais AWS vêm do ambiente (~/.aws/credentials ou variáveis de ambiente)
s3 = boto3.client('s3')

# upload
s3.upload_file(parquet_path, bucket_name, s3_key)
print(f"Arquivo {parquet_path} enviado para s3://{bucket_name}/{s3_key}")

# --- opcional: remover o parquet local após upload ---
os.remove(parquet_path)
