import seaborn as sns

def plot_correlation_matrix(df, num_cols):
    corr = df[num_cols].corr()
    plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", square=True)
    plt.title("Matriz de correlação das variáveis financeiras")
    plt.tight_layout()
    plt.show()
