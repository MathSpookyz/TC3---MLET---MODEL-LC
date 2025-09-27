import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def summary_statistics(df):
    """
    Mostra estatísticas descritivas do DataFrame.
    """
    print(df.describe(include='all'))

def plot_distribution(df, col):
    """
    Plota distribuição de uma coluna numérica.
    """
    plt.figure(figsize=(8,4))
    sns.histplot(df[col], bins=30, kde=True)
    plt.title(f'Distribuição de {col}')
    plt.show()

def plot_correlation_matrix(df):
    """
    Plota matriz de correlação das colunas numéricas.
    """
    plt.figure(figsize=(10,8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Matriz de Correlação")
    plt.show()