import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os

# Caminho absoluto para o diretório do script
BASE_DIR = os.path.dirname(__file__)

# Caminho para o CSV relativo à pasta raiz do projeto
file_path = os.path.join(BASE_DIR, "..", "data", "raw", "lung_cancer_dataset.csv")

# Lê o CSV
df = pd.read_csv(file_path)

def plot_distribution(df, col):
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribuição de {col}')
    plt.show()

def summary_statistics(df):
    """
    Retorna estatísticas descritivas para todas as colunas.
    """
    print("\n===== Estatísticas numéricas =====")
    print(df.describe())
    print("\n===== Estatísticas categóricas =====")
    print(df.describe(include='object'))