from src.data_processing import fetch_financial_metrics
from src.preprocessing import create_preprocessor, preview_transformation
from src.eda import summary_statistics, plot_distribution, plot_correlation_matrix
import pandas as pd
import os
from sklearn.preprocessing import OneHotEncoder

# --- Tickers ---
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META',
    'BRK-B', 'JPM', 'V', 'JNJ', 'WMT', 'UNH', 'LLY', 'XOM', 'MA',
    'PG', 'HD', 'BAC', 'DIS', 'NFLX', 'ADBE', 'CRM', 'CSCO', 'NKE',
    'KO', 'MCD', 'PFE', 'ORCL',
    'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 
    'ABEV3.SA', 'RENT3.SA', 'WEGE3.SA', 'ELET3.SA', 'SUZB3.SA', 
    'B3SA3.SA', 'GGBR4.SA', 'EGIE3.SA', 'HAPV3.SA', 'RDOR3.SA', 
    'TOTS3.SA', 'RADL3.SA', 'PRIO3.SA', 'LREN3.SA', 'MGLU3.SA', 
    'VIVT3.SA', 'SBSP3.SA', 'FLRY3.SA', 'GOAU4.SA', 'CPLE6.SA', 
    'ENGI11.SA', 'SANB11.SA', 'BPAC11.SA', 'CXSE3.SA', 'ASAI3.SA'
]

# --- Fetch financial metrics ---
df = fetch_financial_metrics(tickers)

# --- Preprocessing ---
num_cols = ['roe','debt_to_equity','profit_margins','beta','market_cap',
            'net_income','total_equity','total_debt']

# Garantir que colunas numéricas são float
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

preprocessor = create_preprocessor(num_cols=num_cols)
X_transformed = preview_transformation(preprocessor, df[num_cols])

df_processed = X_transformed.copy()
df_processed['ticker'] = df['ticker'].values  # opcional, para referência

# --- EDA ---
# Estatísticas descritivas apenas das colunas numéricas
summary_statistics(df[num_cols])

# Histogramas apenas das colunas numéricas
for col in num_cols:
    plot_distribution(df, col)

# Correlação apenas das colunas numéricas
plot_correlation_matrix(df[num_cols])
