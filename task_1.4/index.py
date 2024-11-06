import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import umap

diamonds_data = pd.read_csv('task_1.4/diamonds.csv')

enc = OneHotEncoder(handle_unknown='ignore')
enc.fit(diamonds_data[['cut', 'color', 'clarity']])
data_cat_tr = enc.transform(diamonds_data[['cut', 'color', 'clarity']]).toarray()

X_number = diamonds_data.drop(['cut', 'color', 'clarity'], axis=1)
Z = np.hstack((X_number, data_cat_tr))

scaler = StandardScaler()
Z_scaled = scaler.fit_transform(Z)

pca = PCA(n_components=2)
Z_pca = pca.fit_transform(Z_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(Z_pca[:, 0], Z_pca[:, 1], alpha=0.5)
plt.title('PCA: 2 Components')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.show()

pca_full = PCA(n_components=Z_scaled.shape[1])
pca_full.fit(Z_scaled)
explained_variance = pca_full.explained_variance_ratio_

print("Explained variance by each component:")
print(explained_variance)

cumulative_variance = np.cumsum(explained_variance)
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o')
plt.title('Cumulative Explained Variance')
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.grid()
plt.show()



# Применение UMAP
umap_model = umap.UMAP(n_neighbors=15, min_dist=0.1, metric='euclidean')
Z_umap = umap_model.fit_transform(Z_scaled)

plt.figure(figsize=(8, 6))
plt.scatter(Z_umap[:, 0], Z_umap[:, 1], alpha=0.5)
plt.title('UMAP: 2 Components')
plt.xlabel('UMAP Component 1')
plt.ylabel('UMAP Component 2')
plt.show()