"""
clustering.py
Funções utilitárias para rodar DBSCAN em dados financeiros já pré-processados.

Principais funcionalidades:
- compute_k_distance_plot: ajuda a escolher o epsilon (eps) através do k-distance plot.
- fit_dbscan: pipeline que aplica (opcional) StandardScaler, (opcional) PCA e roda DBSCAN.
- describe_clusters: retorna estatísticas agregadas por cluster.
- plot_clusters_2d: visualização em 2D (usa PCA se necessário).
- compute_cluster_metrics: conta clusters, ruídos e calcula silhouette quando aplicável.

Design choices:
- O DBSCAN não "contém" PCA: aplicamos PCA antes do DBSCAN se o usuário desejar.
- Se seus dados já estão escalados (por exemplo: já passaram pelo preprocessor do seu projeto),
  basta chamar fit_dbscan(..., pre_scaled=True).
"""

from typing import Optional, Tuple, Union, Dict
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_k_distance(X: Union[pd.DataFrame, np.ndarray], k: int = 5) -> np.ndarray:
    """
    Calcula as distâncias k-ésimas (k-distance) para cada ponto.
    Retorna as distâncias ordenadas (úteis para plot do "k-distance graph").
    - X: array-like (n_samples, n_features)
    - k: número de vizinhos (usualmente = min_samples)
    """
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    neigh = NearestNeighbors(n_neighbors=k)
    neigh.fit(X_arr)
    distances, _ = neigh.kneighbors(X_arr)
    # distances[:, -1] = distância ao k-ésimo vizinho (aquela que nos interessa)
    kdist = np.sort(distances[:, -1])
    return kdist


def plot_k_distance(X: Union[pd.DataFrame, np.ndarray], k: int = 5, figsize=(8, 4)):
    """
    Plota o k-distance graph para ajudar a escolher eps.
    O "cotovelo" (knee) no gráfico é um bom candidato para eps.
    """
    kdist = compute_k_distance(X, k)
    plt.figure(figsize=figsize)
    plt.plot(np.arange(1, len(kdist) + 1), kdist)
    plt.xlabel("Pontos ordenados")
    plt.ylabel(f"{k}-distance")
    plt.title(f"k-distance plot (k={k})")
    plt.grid(True)
    plt.show()


def _guess_eps_from_kdist(kdist: np.ndarray) -> float:
    """
    Heurística simples para estimar um 'knee' no k-distance.
    Método: procura o índice com maior aumento (diferença primeira derivada).
    Retorna o valor de kdist no índice estimado.
    OBS: é apenas uma sugestão. Conferir via plot_k_distance sempre.
    """
    diffs = np.diff(kdist)
    if len(diffs) == 0:
        return float(kdist.mean())
    knee_idx = int(np.argmax(diffs)) + 1  # +1 porque diff reduz 1 no tamanho
    return float(kdist[knee_idx])


def fit_dbscan(
    X: Union[pd.DataFrame, np.ndarray],
    *,
    pre_scaled: bool = False,
    use_pca: bool = False,
    n_components: int = 2,
    eps: Optional[float] = None,
    min_samples: Optional[int] = None,
    metric: str = "euclidean",
    return_transformed: bool = True
) -> Tuple[np.ndarray, DBSCAN, Dict]:
    """
    Ajusta um DBSCAN aos dados X.
    - X: DataFrame ou ndarray de features (cada linha = uma empresa)
    - pre_scaled: se True, assume que X já foi escalado (não aplica StandardScaler)
    - use_pca: se True, aplica PCA(n_components) e aplica DBSCAN nos componentes
    - n_components: n componentes para PCA (se use_pca=True)
    - eps: raio para DBSCAN. Se None, tentamos estimar via k-distance (heurística).
    - min_samples: se None, será definido como max(5, 2 * n_features) como default
    - metric: métrica de distância para DBSCAN (padrão 'euclidean')
    - return_transformed: se True, retorna também o X efetivamente usado pelo DBSCAN

    Retorna:
      labels: array de rótulos (shape n_samples), ruído = -1
      model: instância ajustada de sklearn.cluster.DBSCAN
      info: dict com chaves 'X_scaled' (se aplicado), 'X_for_clustering', 'scaler', 'pca', 'eps_used', 'min_samples'
    """
    X_orig = X
    X_arr = X.values if isinstance(X, pd.DataFrame) else np.asarray(X)
    n_samples, n_features = X_arr.shape

    # default min_samples: regra prática
    if min_samples is None:
        # regra prática: 2 * n_features (ou pelo menos 5)
        min_samples = max(5, 2 * n_features)

    # Escalonamento (se necessário)
    scaler = None
    if not pre_scaled:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
    else:
        X_scaled = X_arr.copy()

    # PCA opcional (se usar, DBSCAN roda nos componentes)
    pca = None
    if use_pca:
        pca = PCA(n_components=n_components)
        X_for_clustering = pca.fit_transform(X_scaled)
    else:
        X_for_clustering = X_scaled

    # Se eps não foi passado, sugerir/escolher via heurística de k-distance
    if eps is None:
        try:
            # usamos k = min_samples para o k-distance
            kdist = compute_k_distance(X_for_clustering, k=min_samples)
            eps_guess = _guess_eps_from_kdist(kdist)
            # fallback: valor razoável (percentil alto)
            eps_percentile = float(np.percentile(kdist, 90))
            # escolher entre heurística e percentil (retornamos os dois para o usuário)
            eps = float(eps_guess)
            logger.info("eps sugerido (heurística de 'knee') = %.4f ; eps_percentile(90) = %.4f", eps_guess, eps_percentile)
        except Exception as e:
            logger.warning("Falha ao estimar eps via k-distance: %s. Usando eps=0.5 fallback.", e)
            eps = 0.5

    # Ajustar DBSCAN
    db = DBSCAN(eps=eps, min_samples=int(min_samples), metric=metric)
    labels = db.fit_predict(X_for_clustering)

    info = {
        "X_scaled": X_scaled,
        "X_for_clustering": X_for_clustering,
        "scaler": scaler,
        "pca": pca,
        "eps_used": eps,
        "min_samples": min_samples,
        "metric": metric
    }

    if return_transformed:
        return labels, db, info
    else:
        return labels, db, {k: v for k, v in info.items() if k in ('eps_used', 'min_samples', 'metric')}


def compute_cluster_metrics(labels: np.ndarray, X_for_metrics: Union[np.ndarray, pd.DataFrame]) -> Dict:
    """
    Calcula métricas simples de qualidade do clustering:
    - n_clusters (exclui ruído -1), n_noise, silhouette (se aplicável)
    """
    labels = np.asarray(labels)
    n_noise = int((labels == -1).sum())
    cluster_labels = labels[labels != -1]
    n_clusters = len(np.unique(cluster_labels)) if cluster_labels.size > 0 else 0

    metrics = {"n_clusters": n_clusters, "n_noise": n_noise, "n_total": labels.size}

    # silhouette: só faz sentido com >=2 clusters (não contando ruído)
    try:
        if n_clusters >= 2:
            # silhouette exige features com mesmo número de linhas que os rótulos considerados
            X_arr = X_for_metrics if isinstance(X_for_metrics, np.ndarray) else (X_for_metrics.values if isinstance(X_for_metrics, pd.DataFrame) else np.asarray(X_for_metrics))
            # calcular silhouette apenas para pontos rotulados (sem ruído)
            mask = labels != -1
            silhouette = silhouette_score(X_arr[mask], labels[mask])
            metrics["silhouette"] = float(silhouette)
        else:
            metrics["silhouette"] = None
    except Exception as e:
        metrics["silhouette"] = None
        logger.warning("Não foi possível calcular silhouette: %s", e)

    return metrics


def describe_clusters(df: pd.DataFrame, labels: np.ndarray, feature_cols: Optional[list] = None) -> pd.DataFrame:
    """
    Retorna um DataFrame com a média e a contagem por cluster.
    - df: DataFrame original (com as colunas de features originais)
    - labels: rótulos retornados pelo DBSCAN
    - feature_cols: lista de colunas numéricas a serem agregadas; se None, tenta usar todas numéricas
    """
    dfc = df.copy()
    dfc = dfc.reset_index(drop=True)
    dfc["cluster"] = labels
    if feature_cols is None:
        feature_cols = dfc.select_dtypes(include=[np.number]).columns.tolist()
        # remover a coluna cluster se estiver na lista
        feature_cols = [c for c in feature_cols if c != "cluster"]

    agg = dfc.groupby("cluster")[feature_cols].agg(["mean", "median", "std", "count"])
    return agg


def plot_clusters_2d(X: Union[pd.DataFrame, np.ndarray],
                     labels: np.ndarray,
                     *,
                     df: Optional[pd.DataFrame] = None,
                     title: Optional[str] = None,
                     figsize=(8, 6),
                     annotate: bool = False):
    """
    Plota clusters em 2D. Se os dados passados tiverem >2 dimensões, recomenda-se
    passar X reduzido via PCA para 2 componentes.
    - X: (n_samples, 2) ou DataFrame com 2 colunas (componentes)
    - labels: rótulos DBSCAN
    - df: (opcional) DataFrame original para usar como rótulos nos pontos
    """
    X_arr = X if isinstance(X, np.ndarray) else (X.values if isinstance(X, pd.DataFrame) else np.asarray(X))
    if X_arr.shape[1] != 2:
        raise ValueError("plot_clusters_2d requer X com 2 colunas (usar PCA antes para reduzir a 2D).")

    unique_labels = np.unique(labels)
    plt.figure(figsize=figsize)
    colors = plt.cm.get_cmap("tab20", len(unique_labels))

    for idx, lab in enumerate(unique_labels):
        mask = labels == lab
        if lab == -1:
            # ruído: plot em cinza/ preto
            plt.scatter(X_arr[mask, 0], X_arr[mask, 1], c="lightgrey", s=30, label="noise", edgecolor='k', alpha=0.7)
        else:
            plt.scatter(X_arr[mask, 0], X_arr[mask, 1], c=[colors(idx)], s=40, label=f"cluster {lab}", edgecolor='k', alpha=0.8)

    plt.legend()
    plt.title(title or "DBSCAN clusters (2D)")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True)

    if annotate and df is not None:
        # anotar com ticker ou índice (cuidado com poluição visual)
        labels_to_show = df.index.astype(str).tolist()
        for i, txt in enumerate(labels_to_show):
            plt.annotate(txt, (X_arr[i, 0], X_arr[i, 1]), fontsize=6, alpha=0.7)

    plt.show()
