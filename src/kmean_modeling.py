from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

DEFAULT_NUM_COLS = ['roe', 'debt_to_equity', 'profit_margins', 'beta',
                    'market_cap', 'net_income', 'total_equity', 'total_debt']

def calculate_inertia(scaled_features):
    """
    Calcula a inercia e gera o grafico cotovelo do modelo
    """
    inertia = []
    for n_clusters in range(1, 11):
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        kmeans.fit(scaled_features)
        inertia.append(kmeans.inertia_)
        
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, 11), inertia, marker='o')
    plt.title('Método do Cotovelo para o K Ótimo')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')
    plt.show()
    
    
def apply_kmeans_and_visualize(scaled_features, data, optimal_k=3):
    # 1. Garante que os dados estão em formato DataFrame e com nomes de colunas corretos
    if isinstance(data, pd.DataFrame):
        possible_cols = [c for c in DEFAULT_NUM_COLS if c in data.columns]
    else:
        possible_cols = []

    if isinstance(scaled_features, pd.DataFrame):
        features_df = scaled_features.copy()
        if features_df.columns.dtype == object and len(features_df.columns) == len(possible_cols):
            features_df.columns = possible_cols or features_df.columns
    else:
        cols = possible_cols if len(possible_cols) == np.shape(scaled_features)[1] else DEFAULT_NUM_COLS[:np.shape(scaled_features)[1]]
        features_df = pd.DataFrame(scaled_features, columns=cols)

    # 2. Seleciona apenas as colunas numéricas para o agrupamento
    numeric_features = features_df.select_dtypes(include=[np.number])
    if numeric_features.shape[1] == 0:
        raise ValueError("No numeric features available for clustering.")

    # 3. Aplica o KMeans para agrupar os dados
    kmeans_model = KMeans(n_clusters=optimal_k)
    clusters = kmeans_model.fit_predict(numeric_features.values)

    # 4. Monta o DataFrame de resultados, incluindo os dados originais e os escalados
    if isinstance(data, pd.DataFrame):
        results = data.reset_index(drop=True).copy()
    else:
        results = pd.DataFrame(index=range(len(clusters)))

    scaled_prefixed = numeric_features.add_prefix('scaled_').reset_index(drop=True)
    results = pd.concat([results.reset_index(drop=True), scaled_prefixed], axis=1)
    results['Cluster'] = clusters

    # 5. Calcula e mostra as médias dos clusters para ROE e D/E
    summary_cols = [c for c in ['roe', 'debt_to_equity'] if c in results.columns]
    if len(summary_cols) >= 2:
        cluster_means = results.groupby('Cluster')[summary_cols].mean()
    else:
        cluster_means = results.groupby('Cluster')[[f'scaled_roe', f'scaled_debt_to_equity']].mean().rename(columns={
            'scaled_roe': 'roe', 'scaled_debt_to_equity': 'debt_to_equity'
        })

    print("\nMédias dos Clusters (por ROE e D/E quando disponíveis):")
    print(cluster_means)

    # 6. Visualiza os clusters em um gráfico de dispersão
    x_col = 'scaled_roe' if 'scaled_roe' in results.columns else ( 'roe' if 'roe' in results.columns else results.columns[0] )
    y_col = 'scaled_debt_to_equity' if 'scaled_debt_to_equity' in results.columns else ( 'debt_to_equity' if 'debt_to_equity' in results.columns else results.columns[1] )

    plt.figure(figsize=(10, 7))
    sns.scatterplot(x=x_col, y=y_col, hue='Cluster', data=results, palette='viridis', s=100, legend='full')

    # Adiciona os tickers como rótulo nos pontos do gráfico
    ticker_col = None
    for candidate in ['ticker', 'Ticker']:
        if candidate in results.columns:
            ticker_col = candidate
            break

    if ticker_col:
        for _, row in results.iterrows():
            plt.annotate(str(row[ticker_col]), (row[x_col], row[y_col]), xytext=(5,5), textcoords='offset points', fontsize=8,
                         bbox=dict(facecolor='white', alpha=0.6, edgecolor='none'))

    plt.title('Classificação de Empresas (ROE vs D/E) - escalado quando aplicável')
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    return results