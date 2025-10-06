import matplotlib.pyplot as plt

def plot_clusters_2d(X_pca, labels, df=None, title="Clusters DBSCAN (PCA 2D)"):
    plt.figure(figsize=(8, 6))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(range(len(unique_labels)))

    for k, col in zip(unique_labels, colors):
        mask = (labels == k)
        label_name = f"Cluster {k}" if k != -1 else "Ruído / Outlier"
        plt.scatter(
            X_pca[mask, 0], X_pca[mask, 1],
            c=[col], label=label_name, s=60, alpha=0.7, edgecolor='k'
        )

    plt.title(title)
    plt.xlabel("Componente Principal 1")
    plt.ylabel("Componente Principal 2")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
