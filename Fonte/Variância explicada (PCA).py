def plot_explained_variance(pca):
    import numpy as np
    import matplotlib.pyplot as plt

    explained = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=(8,5))
    plt.plot(range(1, len(explained)+1), explained, marker='o')
    plt.title('Variância explicada cumulativa (PCA)')
    plt.xlabel('Número de Componentes')
    plt.ylabel('Variância acumulada')
    plt.grid(True, alpha=0.3)
    plt.show()
