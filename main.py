from src.data_processing import load_and_process_data
from src.preprocessing import create_preprocessor, preview_transformation
from src.eda import plot_distribution, summary_statistics
import os
import sys

# Define BASE_DIR como o diretório onde está o main.py
BASE_DIR = os.path.dirname(__file__)

# Caminho para o dataset dentro da pasta data/raw
file_path = os.path.join(BASE_DIR, "data", "raw", "lung_cancer_dataset.csv")

# Carrega e processa os dados
df = load_and_process_data(file_path)

# Estatísticas descritivas
summary_statistics(df)

# Plot de distribuição de colunas numéricas
for col in ['age','pack_years','exposure_count']:
    plot_distribution(df, col)

# Pasta para salvar o arquivo
output_dir = os.path.join(BASE_DIR, "data", "processed")

df = load_and_process_data(file_path)

num_cols = ['age','pack_years','exposure_count']
ordinal_cols = ['radon_exposure']
cat_cols = ['gender','asbestos_exposure','secondhand_smoke_exposure','copd_diagnosis','alcohol_consumption','family_history']

preprocessor = create_preprocessor(num_cols, cat_cols, ordinal_cols)

X = df.drop(columns=['lung_cancer'])

# Preview transformação
preview_transformation(preprocessor, X)

# Caminho completo do arquivo CSV
output_file = os.path.join(output_dir, "lung_cancer_transformed.csv")

transformed_df = preview_transformation(preprocessor, X)
# Salva o dataframe
transformed_df.to_csv(output_file, index=False)

print(f"Arquivo salvo em: {output_file}")

print("Python path:", sys.path)
print("Current dir:", os.getcwd())