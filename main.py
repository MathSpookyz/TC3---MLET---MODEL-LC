from Fonte.data_processing import fetch_financial_metrics
from Fonte.preprocessing import create_preprocessor, preview_transformation
from Fonte.eda import summary_statistics, plot_distribution, plot_correlation_matrix
from Fonte.pca_utils import fit_pca, plot_explained_variance, plot_pca_2d, biplot
from Fonte.clustering import fit_dbscan, compute_cluster_metrics, describe_clusters, plot_clusters_2d
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import pandas as pd
import time
import os, sys
## garantir X_pca como DataFrame/ndarray
import numpy as np, pandas as pd
# garantir X_pca com 2 colunas (PC1, PC2)
import pandas as pd, numpy as np
from math import isfinite

print("cwd:", os.getcwd())
print("__file__:", __file__)
print("Lista de arquivos no cwd:", os.listdir(os.getcwd()))
print("Numero de linhas no df (se existir):", "df not defined yet" if 'df' not in globals() else len(df))

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
print("1) Iniciando pré-processamento...")
num_cols = ['roe','debt_to_equity','profit_margins','beta','market_cap',
            'net_income','total_equity','total_debt']

# Garantir que colunas numéricas são float
for col in num_cols:
    df[col] = pd.to_numeric(df[col], errors='coerce')
    df[col].fillna(df[col].median(), inplace=True)

# Antes de criar o preprocessor, converta em numérico sem fillna:
for col in num_cols:
    df.loc[:, col] = pd.to_numeric(df[col], errors='coerce')
    
# (assume que df e num_cols já existem)
preprocessor = create_preprocessor(num_cols=num_cols)
try:
    transformed = preview_transformation(preprocessor, df[num_cols])
    # preview_transformation pode retornar numpy.ndarray ou pd.DataFrame
    if isinstance(transformed, np.ndarray):
        # Se for ndarray usamos os nomes originais num_cols (assume 1:1)
        if transformed.shape[1] != len(num_cols):
            raise ValueError(f"preview_transformation retornou ndarray com shape {transformed.shape} diferente de num_cols ({len(num_cols)})")
        X_transformed = pd.DataFrame(transformed, columns=num_cols, index=df.index)
        print("preview_transformation retornou ndarray — convertido para DataFrame.")
    elif isinstance(transformed, pd.DataFrame):
        X_transformed = transformed.copy()
        # se colunas não existirem, renomear para num_cols (opcional)
        if list(X_transformed.columns) != num_cols and X_transformed.shape[1] == len(num_cols):
            X_transformed.columns = num_cols
            print("Renomeadas colunas do DataFrame transformado para num_cols.")
    else:
        raise TypeError(f"preview_transformation retornou tipo inesperado: {type(transformed)}")
except Exception as e:
    print("ERRO no preview_transformation:", e)
    raise
print("Pré-processamento finalizado. Linhas:", X_transformed.shape[0], "Colunas:", X_transformed.shape[1])
# opcional: inspecionar primeiras linhas
print(X_transformed.head())

df_processed = X_transformed.copy()
df_processed['ticker'] = df['ticker'].values  # opcional, para referência

# ---PCA e DBSCAN---
# PCA
print("2) Aplicando PCA (retenção 95%)...")
try:
    pca_info = fit_pca(X_transformed, variance_threshold=0.95, pre_scaled=True)
    X_pca = pca_info["X_pca"]
    print("PCA pronto — componentes:", pca_info["n_components_used"])
except Exception as e:
    print("ERRO ao rodar PCA:", e)
    raise

# DBSCAN
print("3) Rodando DBSCAN...")
try:
    labels, model, info = fit_dbscan(X_pca, pre_scaled=True, use_pca=False, eps=None, min_samples=None)
    print("DBSCAN finalizado. clusters únicos:", sorted(set(labels)))
except Exception as e:
    print("ERRO ao rodar DBSCAN:", e)
    raise

# ======= procura automática por parâmetros DBSCAN para obter +2 clusters =======
import numpy as np
from math import isfinite
print("\n--- Calibrando DBSCAN para obter +2 clusters (busca automática) ---")

# 1) número atual de clusters (sem contar ruído)
metrics_now = compute_cluster_metrics(labels, X_pca)
current_n = metrics_now.get("n_clusters", 0)
target_n = current_n + 2
print(f"Clusters atuais: {current_n}. Objetivo: {target_n} clusters.")

# 2) definir grade de busca (ajuste conforme necessário)
# eps em escala de 0.05 a 1.2 é um ponto de partida razoável para dados já padronizados/PCA.
eps_values = np.linspace(0.05, 1.2, 24)   # 24 valores entre 0.05 e 1.2
min_samples_values = [6, 7, 8, 9, 10]     # testar várias densidades mínimas

candidates = []

for eps in eps_values:
    for ms in min_samples_values:
        try:
            lbls, model_try, info_try = fit_dbscan(X_pca, pre_scaled=True, use_pca=False, eps=float(eps), min_samples=int(ms))
            m = compute_cluster_metrics(lbls, info_try["X_for_clustering"])
            if m["n_clusters"] == target_n:
                # compute silhouette (já calculado em compute_cluster_metrics se aplicável)
                sil = m.get("silhouette", None)
                candidates.append({
                    "eps": float(eps),
                    "min_samples": int(ms),
                    "n_noise": m["n_noise"],
                    "silhouette": sil if (sil is not None and isfinite(sil)) else -999,
                    "labels": lbls,
                    "model": model_try
                })
        except Exception as e:
            # ignorar combinações que lançarem erro (ex.: num neigh too large)
            continue

print(f"Encontradas {len(candidates)} combinações que produzem {target_n} clusters.")

# 3) escolher melhor candidato (priorizar silhouette, depois menos ruído)
best = None
if candidates:
    # ordenar por silhouette desc, n_noise asc
    candidates_sorted = sorted(candidates, key=lambda c: (c["silhouette"], -c["n_noise"]), reverse=True)
    best = candidates_sorted[0]
    print("Melhor candidato encontrado:", {"eps": best["eps"], "min_samples": best["min_samples"], "silhouette": best["silhouette"], "n_noise": best["n_noise"]})
    # aplicar labels escolhidos
    labels_new = best["labels"]
    df_processed["cluster"] = labels_new
    print("Clusters atualizados no df_processed com os novos rótulos.")
    # opcional: sobrescrever labels e model para plot posteriores
    labels = labels_new
    model = best["model"]
else:
    print("Nenhuma combinação na grade produziu exatamente o número alvo de clusters.")
    print("Opções: (1) ampliar grade de eps/min_samples; (2) usar KMeans para forçar número de clusters.")

# 4) se um candidato válido foi encontrado, plotar resultado
if best:
    # se X_pca tiver múltiplas colunas, usar as duas primeiras para plotar
    X_plot = X_pca.iloc[:, :2].values if hasattr(X_pca, "iloc") else np.asarray(X_pca)[:, :2]
    plot_clusters_2d(X_plot, labels, df=df_processed, title=f"DBSCAN tuned -> {target_n} clusters (eps={best['eps']}, ms={best['min_samples']})")


# --- EDA ---
# Estatísticas descritivas apenas das colunas numéricas
summary_statistics(df[num_cols])

# Histogramas apenas das colunas numéricas
for col in num_cols:
    plot_distribution(df, col)

# X_pca pode ser DataFrame (recomendado) ou ndarray
if isinstance(X_pca, pd.DataFrame):
    if X_pca.shape[1] < 2:
        raise ValueError("X_pca tem menos de 2 componentes — não é possível plotar 2D.")
    X_plot = X_pca.iloc[:, :2].values  # pega PC1 e PC2
else:
    X_pca = np.asarray(X_pca)
    if X_pca.ndim != 2 or X_pca.shape[1] < 2:
        raise ValueError("X_pca precisa ser 2D e ter ao menos 2 colunas.")
    X_plot = X_pca[:, :2]

# agora chama a função que exige 2 colunas
plot_clusters_2d(X_plot, labels, df=df_processed, title="Clusters DBSCAN (PC1 vs PC2)")

# se X_pca for DataFrame:
if isinstance(X_pca, pd.DataFrame):
    X_plot = X_pca.iloc[:, :2].values   # pega PC1 e PC2
else:
    X_arr = np.asarray(X_pca)
    X_plot = X_arr[:, :2]

# então chama a função que exige 2 colunas
plot_clusters_2d(X_plot, labels, df=df_processed, title="Clusters DBSCAN (PC1 vs PC2)")


#Plot Gráfico dos Clusters em 2D (PCA)
plot_clusters_2d(X_pca, labels, df=df_processed)

#Plot Médias por cluster (gráfico de barras)
plot_cluster_means(df_processed, num_cols)

#Plot Variância explicada (PCA)
plot_explained_variance(pca)

#Plot Matriz de correlação das variáveis originais
plot_correlation_matrix(df, num_cols)

