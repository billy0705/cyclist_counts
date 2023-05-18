import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv('cyclist_counts.csv', delimiter=';')

df['datetime'] = df['date'] + ' ' + df['time']

df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S')
df.dropna(axis=0, inplace=True)

# df.drop(columns=['date', 'time'], inplace=True)

print(df.info())

groups = df.groupby(pd.Grouper(key='datetime', freq='D'))
data = []

col_name = df.drop(columns=['datetime', 'date', 'time']).columns
print(col_name)
for _, group in groups:
    matrix = group.drop(columns=['datetime', 'date', 'time']).to_numpy()
    # matrix = group["fleher deich ost stromaufwaerts"].to_numpy()
    # matrix = group["fleher deich west stromabwaerts"].to_numpy()
    # matrix = group["okb nord"].to_numpy()
    # matrix = group["okb sued"].to_numpy()
    # print(len(matrix))
    if len(matrix) == 96:
        data.append(matrix)
    else:
        pass
        # print(groups.info())


dimensions = set([matrix.shape[0] for matrix in data])

if len(dimensions) > 1:
    raise ValueError("dimensions error")

X = np.array(data)
# print(X)

from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# k_values = range(2, 10)
# inertias = []
# silhouette_scores_k = []
# silhouette_scores_d = []

# for k in k_values:
#     print("k=", k)
#     n_clusters = k
#     max_iter = 20
#     n_init = 20
#     random_state = 42

#     # Clustering model
#     kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=random_state)

#     # data Clustering
#     kmeans.fit(X.reshape(364, -1))
#     inertias.append(kmeans.inertia_)
#     silhouette_avg_k = silhouette_score(X.reshape(364, -1), kmeans.labels_)
#     silhouette_scores_k.append(silhouette_avg_k)

# plt.plot(k_values, inertias, 'bx-')
# plt.xlabel('K Value')
# plt.ylabel('Inertia')
# plt.title('Elbow Method')
# plt.figure()
# plt.plot(k_values, silhouette_scores_k, 'bx-')
# plt.xlabel('K Value')
# plt.ylabel('Silhouette Coefficient')
# plt.title('Silhouette Coefficient Method')
# plt.show()

# setting Clustering config
n_clusters = 2
max_iter = 20
n_init = 20
random_state = 42

# Clustering model
kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=random_state)


# X.reshape(-1, X.shape[-1])
# data Clustering
kmeans.fit(X.reshape(364, -1))

# get Clustering results
labels = kmeans.labels_
print(labels)

clusters = []
for i in range(n_clusters):
    cluster = X[labels == i]
    clusters.append(cluster)

colors = ['r', 'g', 'b', 'yellow', 'pink', 'brown', 'gray', 'purple', 'orange', 'violet']


fig, axs = plt.subplots(len(col_name), n_clusters, figsize=(15, 6))

t = range(X.shape[1])
for i in range(len(col_name)):
    for j in range(n_clusters):
        cluster = clusters[j]
        for k in range(len(cluster)):
            axs[i,j].plot(t, cluster[k][:, i], c=colors[j])
        axs[0, j].set_title(f'Cluster {j+1}')
    axs[i, 0].set_ylabel(col_name[i], wrap=True, rotation=1, labelpad=50)

plt.show()