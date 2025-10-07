from src.kmean_modeling import calculate_inertia, apply_kmeans_and_visualize
from src.data_processing import fetch_financial_metrics
from src.preprocessing import create_preprocessor, preview_transformation
from src.eda import summary_statistics, plot_distribution, plot_correlation_matrix
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder
import os

os.environ["LOKY_MAX_CPU_COUNT"] = "4"  # Replace 4 with your actual number of physical cores

# --- Tickers ---
tickers = [
    # 'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    # 'BRK-B', 'JPM', 'V', 'JNJ', 'WMT', 'UNH', 'LLY', 'XOM', 'MA',
    # 'PG', 'HD', 'BAC', 'DIS', 'NFLX', 'ADBE', 'CRM', 'CSCO', 'NKE',
    # 'KO', 'MCD', 'PFE', 'ORCL',
    'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 
    'ABEV3.SA', 'RENT3.SA', 'WEGE3.SA', 'ELET3.SA', 'SUZB3.SA', 
    'B3SA3.SA', 'GGBR4.SA', 'EGIE3.SA', 'HAPV3.SA', 'RDOR3.SA', 
    'TOTS3.SA', 'RADL3.SA', 'PRIO3.SA', 'LREN3.SA', 'MGLU3.SA', 
    'VIVT3.SA', 'SBSP3.SA', 'FLRY3.SA', 'GOAU4.SA', 'CPLE6.SA', 
    'ENGI11.SA', 'SANB11.SA', 'BPAC11.SA', 'CXSE3.SA', 'ASAI3.SA'
]

# --- Fetch financial metrics ---
data = fetch_financial_metrics(tickers)

# --- Preprocessing ---
num_cols = ['roe','debt_to_equity','profit_margins','beta','market_cap',
            'net_income','total_equity','total_debt']

# Garantir que colunas numéricas são float
for col in num_cols:
    data[col] = pd.to_numeric(data[col], errors='coerce')
    data[col].fillna(data[col].median(), inplace=True)

preprocessor = create_preprocessor(num_cols=num_cols)
scaled_features = preview_transformation(preprocessor, data[num_cols])

df_processed = pd.DataFrame(scaled_features)
df_processed['ticker'] = data['ticker'].values  # opcional, para referência

# --- EDA ---
# Estatísticas descritivas apenas das colunas numéricas
# summary_statistics(df[num_cols])

# Histogramas apenas das colunas numéricas
# for col in num_cols:
#     plot_distribution(data, col)

# Correlação apenas das colunas numéricas
# plot_correlation_matrix(data[num_cols])

# calculate_inertia(scaled_features=scaled_features)
results = apply_kmeans_and_visualize(scaled_features, data, optimal_k=6)

# Show the head of each cluster
for cluster_id in sorted(results['Cluster'].unique()):
    print(f"\nCluster {cluster_id}:")
    print(results[results['Cluster'] == cluster_id].head())