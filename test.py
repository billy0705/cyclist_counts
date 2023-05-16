import pandas as pd
import numpy as np

df = pd.read_csv('cyclist_counts.csv', delimiter=';')

df['datetime'] = df['date'] + ' ' + df['time']

df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S')
df.dropna(axis=0, inplace=True)

# 刪除原始的日期和時間欄位
df.drop(columns=['date', 'time'], inplace=True)

print(df.info())

groups = df.groupby(pd.Grouper(key='datetime', freq='D'))
data = []
for _, group in groups:
    matrix = group['okb nord'].to_numpy()
    print(len(matrix))
    if len(matrix) == 96:
        matrix = matrix.reshape((1, 96))
        data.append(matrix)
    else:
        pass
        # print(groups.info())


dimensions = set([matrix.shape[1] for matrix in data])

if len(dimensions) > 1:
    raise ValueError("數據矩陣的維度不一致")
# 將每天的數據作為一個樣本，將這些樣本轉換成 numpy array
X = np.array(data)
# print(X)

from sklearn.cluster import KMeans

# 設置 Clustering 的參數
n_clusters = 3
max_iter = 10
n_init = 10
random_state = 42

# 定義 Clustering 模型
kmeans = KMeans(n_clusters=n_clusters, max_iter=max_iter, n_init=n_init, random_state=random_state)

# 將數據做 Clustering
kmeans.fit(X.reshape(-1, X.shape[-1]))

# 獲得 Clustering 結果
labels = kmeans.labels_
print(labels)