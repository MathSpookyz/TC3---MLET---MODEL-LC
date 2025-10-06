import seaborn as sns
import matplotlib.pyplot as plt

def plot_cluster_means(df, num_cols):
    cluster_means = df.groupby('cluster')[num_cols].mean()
    cluster_means.T.plot(kind='bar', figsize=(10,6))
    plt.title("Médias das variáveis por cluster")
    plt.ylabel("Valor médio padronizado")
    plt.xlabel("Variável")
    plt.legend(title="Cluster")
    plt.grid(axis='y', linestyle='--', alpha=0.6)
    plt.tight_layout()
    plt.show()
