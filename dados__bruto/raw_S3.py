import pandas as pd
import boto3
import os
import yfinance as yf
from datetime import datetime

# --- CONFIGURAÇÕES ---
tickers = [
    'AAPL', 'MSFT', 'GOOGL', 'GOOG', 'AMZN', 'NVDA', 'TSLA', 'META',
    'BRK-B', 'JPM', 'V', 'JNJ', 'WMT', 'UNH', 'LLY', 'XOM', 'MA',
    'PG', 'HD', 'BAC', 'DIS', 'NFLX', 'ADBE', 'CRM', 'CSCO', 'NKE',
    'KO', 'MCD', 'PFE', 'ORCL',
    'VALE3.SA', 'PETR4.SA', 'ITUB4.SA', 'BBDC4.SA', 'BBAS3.SA', 
    'ABEV3.SA', 'RENT3.SA', 'WEGE3.SA', 'ELET3.SA', 'SUZB3.SA', 
    'B3SA3.SA', 'GGBR4.SA', 'EGIE3.SA', 'HAPV3.SA', 'RDOR3.SA', 
    'TOTS3.SA', 'RADL3.SA', 'PRIO3.SA', 'LREN3.SA', 'MGLU3.SA', 
    'VIVT3.SA', 'SBSP3.SA', 'FLRY3.SA', 'GOAU4.SA', 'CPLE6.SA', 
    'ENGI11.SA', 'SANB11.SA', 'BPAC11.SA', 'CXSE3.SA', 'ASAI3.SA']

bucket_name = "dados-raw-finance"
s3_key = f"data/finance_metrics_{datetime.today().date()}.parquet"
parquet_path = "finance_metrics.parquet"

# --- COLETA DE DADOS ---
data_list = []

for t in tickers:
    ticker = yf.Ticker(t)

    # --- Informações gerais ---
    info = ticker.info
    roe = info.get('returnOnEquity')
    debt_to_equity = info.get('debtToEquity')
    profit_margins = info.get('profitMargins')
    beta = info.get('beta')
    market_cap = info.get('marketCap')

    # --- Dados financeiros detalhados ---
    income_statement = ticker.financials
    balance_sheet = ticker.balance_sheet

    # Cuidado com índices que podem não existir
    try:
        net_income = income_statement.loc['Net Income'].iloc[0]
    except Exception:
        net_income = None

    try:
        total_equity = balance_sheet.loc['Total Equity Gross Minority Interest'].iloc[0]
    except Exception:
        total_equity = None

    try:
        total_debt = balance_sheet.loc['Total Liabilities Net Minority Interest'].iloc[0]
    except Exception:
        total_debt = None

    data_list.append({
        'ticker': t,
        'roe': roe,
        'debt_to_equity': debt_to_equity,
        'profit_margins': profit_margins,
        'beta': beta,
        'market_cap': market_cap,
        'net_income': net_income,
        'total_equity': total_equity,
        'total_debt': total_debt
    })

df_metrics = pd.DataFrame(data_list)
print(df_metrics)

# --- SALVAR EM PARQUET ---
df_metrics.to_parquet(parquet_path, engine='pyarrow', compression='snappy', index=False)

# --- UPLOAD PARA S3 ---
s3 = boto3.client('s3')
s3.upload_file(parquet_path, bucket_name, s3_key)
print(f"Arquivo {parquet_path} enviado para s3://{bucket_name}/{s3_key}")

# --- LIMPAR ARQUIVO LOCAL ---
os.remove(parquet_path)
