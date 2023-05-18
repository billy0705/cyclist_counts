import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


df = pd.read_csv('cyclist_counts.csv', delimiter=';')

missing_rows = df[df.isnull().any(axis=1)]
print("Missing row (before):")
print(missing_rows)

for index, row in missing_rows.iterrows():
    df.loc[index] = df.loc[index].fillna(0)
    # print(row)
    if row['time'] == '03:00:00':
        df.loc[index, 'time'] = '02:00:00'
    if row['time'] == '03:15:00':
        df.loc[index, 'time'] = '02:15:00'
    if row['time'] == '03:30:00':
        df.loc[index, 'time'] = '02:30:00'
    if row['time'] == '03:45:00':
        df.loc[index, 'time'] = '02:45:00'

missing_rows_after = df[df.isnull().any(axis=1)]
print("Missing row (after):")
print(missing_rows_after)

print(df.info())

df['datetime'] = df['date'] + ' ' + df['time']

df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df.dropna(axis=0, inplace=True)

# df.drop(columns=['date', 'time'], inplace=True)

print(df.info())

############### boxplot
selected_data = df[df['time'] == '08:00:00'].drop(columns=['time', 'datetime'])

df_melted = pd.melt(selected_data[['fleher deich ost stromaufwaerts', 'fleher deich west stromabwaerts', 'okb nord', 'okb sued']], var_name='Columns', value_name='Counts')
plt.figure(figsize=(15, 6))
ax = sns.boxplot(x='Columns', y='Counts', data=df_melted)
ax.set_xticklabels(ax.get_xticklabels(), wrap=True)
title_name = "Data distribution in time = " + '08:00:00'
ax.set_title(title_name)
# plt.show()

############### month

groups = df[df['time'] == '08:00:00'].groupby(pd.Grouper(key='datetime', freq='M'))
col_name = df.drop(columns=['datetime', 'date', 'time']).columns
print(col_name)
data = []
for _, group in groups:
    matrix = group.drop(columns=['datetime', 'date', 'time']).sum().to_numpy()
    data.append(matrix)


dimensions = set([matrix.shape[0] for matrix in data])

if len(dimensions) > 1:
    print("dimensions error")

data = np.array(data)

print(data.shape)
months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
x = np.arange(len(months))

width = 0.2

fig, ax = plt.subplots()
rects1 = ax.bar(x - 1.5 * width, data[:, 0], width, label=col_name[0])
rects2 = ax.bar(x - 0.5 * width, data[:, 1], width, label=col_name[1])
rects3 = ax.bar(x + 0.5 * width, data[:, 2], width, label=col_name[2])
rects4 = ax.bar(x + 1.5 * width, data[:, 3], width, label=col_name[3])

ax.set_xticks(x)
ax.set_xticklabels(months)
ax.set_ylabel("Total Counts")
ax.legend()
# plt.show()


cross_tab = pd.crosstab(selected_data['fleher deich ost stromaufwaerts'], selected_data['fleher deich west stromabwaerts'])

# 打印列聯表
print(cross_tab)
plt.figure()
# cross_tab.plot(kind='bar', stacked=True)
sns.heatmap(cross_tab, annot=True, cmap='YlGnBu')

# 添加標籤和標題
# plt.xlabel('Category')
# plt.ylabel('Count')
plt.title('Cross Tabulation')

# 顯示圖表
plt.show()