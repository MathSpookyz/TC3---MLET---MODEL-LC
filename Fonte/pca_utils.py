"""
pca_utils.py

Utilitários para aplicar e interpretar PCA em datasets tabulares (pandas DataFrame).
Design:
- Funciona bem integrado a um pipeline: aceita dados já escalados (pre_scaled=True)
  ou aplica StandardScaler internamente (pre_scaled=False).
- Permite escolher número de componentes diretamente (n_components) ou por
  nível de variância retida (variance_threshold entre 0 e 1).
- Retorna DataFrames com índices preservados e nomes de colunas "PC1", "PC2", ...
"""

from typing import Optional, Tuple, Dict, Any, List
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _ensure_dataframe(X) -> Tuple[pd.DataFrame, List[str]]:
    """
    Garante que X seja um DataFrame.
    Se X for ndarray, converte para DataFrame com nomes genéricos 'feat_0', 'feat_1', ...
    Retorna (df, feature_names).
    """
    if isinstance(X, pd.DataFrame):
        df = X.copy()
        feature_names = df.columns.tolist()
    else:
        arr = np.asarray(X)
        feature_names = [f"feat_{i}" for i in range(arr.shape[1])]
        df = pd.DataFrame(arr, columns=feature_names)
    return df, feature_names


def fit_pca(
    X,
    *,
    n_components: Optional[int] = None,
    variance_threshold: Optional[float] = None,
    pre_scaled: bool = False,
    scale: bool = True,
    random_state: Optional[int] = None
) -> Dict[str, Any]:
    """
    Ajusta um PCA aos dados X (DataFrame ou ndarray).

    Parâmetros:
    - X: pd.DataFrame ou np.ndarray (linhas = amostras, colunas = features).
    - n_components: int opcional. Número de componentes a manter.
    - variance_threshold: float opcional entre 0 e 1. Se definido, escolhe componentes
                         necessários para reter essa fração de variância (ex.: 0.95).
                         Não combine com n_components (se ambos definidos, n_components tem prioridade).
    - pre_scaled: se True, assume que X já foi escalado (não aplica StandardScaler).
    - scale: se pre_scaled=False e scale=True, aplica StandardScaler antes do PCA.
    - random_state: para reproduzibilidade (passa para PCA).

    Retorna dicionário com chaves:
    - 'pca': objeto sklearn.decomposition.PCA ajustado
    - 'scaler': StandardScaler ajustado (ou None se pre_scaled=True ou scale=False)
    - 'X_scaled': numpy array escalado usado no fit
    - 'X_pca': pd.DataFrame com colunas ['PC1', 'PC2', ...] (index preservado)
    - 'explained_variance_ratio_': np.ndarray (por componente)
    - 'cumulative_variance': np.ndarray (cumulativa)
    - 'n_components_used': int
    - 'feature_names': lista original de features
    """

    # Garantir DataFrame e nomes das features
    df, feature_names = _ensure_dataframe(X)
    X_arr = df.values
    n_samples, n_features = X_arr.shape

    if n_components is not None and variance_threshold is not None:
        logger.warning("Foi passado n_components e variance_threshold. Ignorando variance_threshold e usando n_components.")

    # Escalonamento (quando necessário)
    scaler = None
    if not pre_scaled and scale:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_arr)
    else:
        X_scaled = X_arr.copy()

    # Decidir número de componentes
    if n_components is None and variance_threshold is None:
        # padrão: manter todos (máximo possível)
        n_components_to_use = min(n_features, n_samples)
    elif n_components is not None:
        n_components_to_use = int(n_components)
    else:  # variance_threshold foi definido
        if not (0 < variance_threshold <= 1.0):
            raise ValueError("variance_threshold deve estar entre 0 e 1 (ex.: 0.95).")
        # sklearn aceita float n_components representando proporção; usamos esse recurso
        pca_temp = PCA(n_components=variance_threshold, random_state=random_state)
        pca_temp.fit(X_scaled)
        # sklearn define n_components_ como o número real usado
        n_components_to_use = pca_temp.n_components_

    # Ajustar PCA com n_components_to_use
    pca = PCA(n_components=n_components_to_use, random_state=random_state)
    X_pca_arr = pca.fit_transform(X_scaled)

    # Construir DataFrame de componentes com nomes PC1, PC2, ...
    pc_cols = [f"PC{i+1}" for i in range(X_pca_arr.shape[1])]
    X_pca_df = pd.DataFrame(X_pca_arr, columns=pc_cols, index=df.index)

    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)

    info = {
        "pca": pca,
        "scaler": scaler,
        "X_scaled": X_scaled,
        "X_pca": X_pca_df,
        "explained_variance_ratio_": explained,
        "cumulative_variance": cumulative,
        "n_components_used": pca.n_components_,
        "feature_names": feature_names
    }
    logger.info("PCA ajustado: %d componentes retidos (%.2f%% da variância total acumulada).",
                info["n_components_used"], 100.0 * info["cumulative_variance"][-1])
    return info


def plot_explained_variance(pca_obj, *, figsize=(8, 4), show_cumulative: bool = True):
    """
    Plota a variância explicada (scree) e opcionalmente a variância acumulada.
    - pca_obj: objeto PCA já ajustado
    """
    var_ratio = pca_obj.explained_variance_ratio_
    cumvar = np.cumsum(var_ratio)
    n = len(var_ratio)
    plt.figure(figsize=figsize)
    plt.bar(range(1, n + 1), var_ratio, alpha=0.7, label='Explained variance (individual)')
    if show_cumulative:
        plt.step(range(1, n + 1), cumvar, where='mid', color='red', label='Cumulative variance')
        plt.axhline(y=0.95, color='gray', linestyle='--', linewidth=0.6)  # linha guia
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Ratio')
    plt.title('Scree plot / Explained variance per component')
    plt.legend()
    plt.grid(True)
    plt.show()


def get_feature_loadings(pca_obj, feature_names: List[str]) -> pd.DataFrame:
    """
    Calcula e retorna as cargas (loadings) das features sobre cada componente.
    - loadings = components_.T * sqrt(explained_variance_)
    Retorna DataFrame com index = feature_names e colunas = ['PC1', 'PC2', ...]
    """
    comps = pca_obj.components_  # shape (n_components, n_features)
    ev = pca_obj.explained_variance_  # variância por componente (não ratio)
    # loadings shape (n_features, n_components)
    loadings = (comps.T * np.sqrt(ev))
    pc_cols = [f"PC{i+1}" for i in range(loadings.shape[1])]
    df_load = pd.DataFrame(loadings, index=feature_names, columns=pc_cols)
    return df_load


def plot_pca_2d(X_pca_df: pd.DataFrame, labels: Optional[np.ndarray] = None, *,
                title: Optional[str] = None, figsize=(7, 6), annotate: bool = False, df_labels: Optional[pd.DataFrame] = None):
    """
    Plota scatter 2D usando as duas primeiras colunas de X_pca_df (PC1, PC2).
    - labels: array-like com rótulos para colorir (por exemplo, cluster labels)
    - annotate: se True e df_labels dado, anota pontos com index/coluna especificada
    """
    if X_pca_df.shape[1] < 2:
        raise ValueError("X_pca_df deve ter pelo menos 2 componentes para plot 2D.")
    x = X_pca_df.iloc[:, 0].values
    y = X_pca_df.iloc[:, 1].values
    plt.figure(figsize=figsize)

    if labels is None:
        plt.scatter(x, y, s=40, alpha=0.8)
    else:
        uniq = np.unique(labels)
        cmap = plt.cm.get_cmap("tab10", len(uniq))
        for i, u in enumerate(uniq):
            mask = labels == u
            label_name = f"cluster {u}" if u != -1 else "noise"
            plt.scatter(x[mask], y[mask], s=40, alpha=0.8, label=label_name, color=cmap(i))

        plt.legend()

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title(title or "PCA (PC1 vs PC2)")
    plt.grid(True)

    if annotate and df_labels is not None:
        # df_labels pode ser um DataFrame com coluna 'label' ou usar index
        for i, txt in enumerate(df_labels.index.astype(str)):
            plt.annotate(txt, (x[i], y[i]), fontsize=6, alpha=0.7)

    plt.show()


def biplot(X_pca_df: pd.DataFrame, pca_obj: PCA, feature_names: List[str], *,
           scale_arrows: float = 1.0, top_features: Optional[int] = None, figsize=(9, 7)):
    """
    Biplot 2D: scatter das amostras nas duas primeiras componentes + vetores (arrows)
    representando a contribuição das features nas componentes.
    - top_features: se definido, desenha apenas as top N features por magnitude em PC1+PC2.
    - scale_arrows: fator de escala para as setas (ajuste para visibilidade).
    """
    if X_pca_df.shape[1] < 2:
        raise ValueError("X_pca_df deve ter pelo menos 2 componentes para biplot.")

    x = X_pca_df.iloc[:, 0].values
    y = X_pca_df.iloc[:, 1].values

    # utilizar componentes (2 x n_features) -> transposta para (n_features x 2)
    comps = pca_obj.components_[:2, :].T  # shape (n_features, 2)
    # normalizar vetores para melhor visualização (opcional)
    vecs = comps * scale_arrows

    # selecionar top features se pedido
    if top_features is not None:
        mag = np.linalg.norm(vecs, axis=1)
        top_idx = np.argsort(mag)[-top_features:]
    else:
        top_idx = np.arange(vecs.shape[0])

    plt.figure(figsize=figsize)
    plt.scatter(x, y, s=30, alpha=0.8)
    # plotar vetores
    for i in top_idx:
        plt.arrow(0, 0, vecs[i, 0], vecs[i, 1],
                  head_width=0.03 * np.max(np.abs([x, y])), head_length=0.03 * np.max(np.abs([x, y])),
                  linewidth=1.0, color='r', alpha=0.8)
        plt.text(vecs[i, 0] * 1.05, vecs[i, 1] * 1.05, feature_names[i], color='r', fontsize=9)

    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.title("Biplot (PC1 vs PC2)")
    plt.grid(True)
    plt.axhline(0, color='grey', linewidth=0.5)
    plt.axvline(0, color='grey', linewidth=0.5)
    plt.show()


def transform_new_data(pca_obj: PCA, scaler: Optional[StandardScaler], X_new) -> np.ndarray:
    """
    Transforma novos dados X_new usando o scaler e o pca já ajustados.
    Retorna array transformado (componentes principais).
    """
    df_new, _ = _ensure_dataframe(X_new)
    X_arr = df_new.values
    if scaler is not None:
        X_scaled = scaler.transform(X_arr)
    else:
        X_scaled = X_arr
    X_pca_new = pca_obj.transform(X_scaled)
    return X_pca_new


def reconstruct_approximation(pca_obj: PCA, scaler: Optional[StandardScaler], X_pca) -> np.ndarray:
    """
    Reconstrói aproximação dos dados no espaço original a partir de X_pca (componentes).
    Aplica inverse_transform do PCA e, se houver scaler, inverse_transform do scaler.
    Retorna numpy array com forma (n_samples, n_features_original).
    """
    X_approx_scaled = pca_obj.inverse_transform(X_pca)
    if scaler is not None:
        X_approx = scaler.inverse_transform(X_approx_scaled)
    else:
        X_approx = X_approx_scaled
    return X_approx
