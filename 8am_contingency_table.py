import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import chi2_contingency

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

df['fleher deich ost stromaufwaerts'] = df['fleher deich ost stromaufwaerts'].astype(int)
df['fleher deich west stromabwaerts'] = df['fleher deich west stromabwaerts'].astype(int)
df['okb nord'] = df['okb nord'].astype(int)
df['okb sued'] = df['okb sued'].astype(int)

df['datetime'] = df['date'] + ' ' + df['time']

# df['total'] = df['fleher deich ost stromaufwaerts'] + df['fleher deich west stromabwaerts'] + df['okb nord'] + df['okb sued']

df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S')
# df['year'] = df['datetime'].dt.year
# df['month'] = df['datetime'].dt.month
# df['day'] = df['datetime'].dt.day
# df.dropna(axis=0, inplace=True)

col_name = df.drop(columns=['datetime', 'date', 'time']).columns
df.drop(columns=['date', 'time'], inplace=True)
print(df.info())

# groups = df.groupby(pd.Grouper(key='datetime', freq='M'))
# data = []

# col_name = df.drop(columns=['datetime', 'date', 'time']).columns
# print(col_name)
# for _, group in groups:
#     matrix = group.drop(columns=['datetime', 'date', 'time']).sum().to_numpy()
#     data.append(matrix)

pivot_table = pd.pivot_table(df.drop(columns=['datetime']), index=df['datetime'].dt.to_period('M'), aggfunc='sum')

# 計算每個月份的總和
row_sum = pd.DataFrame(pivot_table.sum(axis=1), columns=['Total'])
pivot_table = pd.concat([pivot_table, row_sum], axis=1)
pivot_table['Percentage'] = (pivot_table['Total'] / pivot_table['Total'].sum() * 100).round(2)

# 計算每個站點的總和
column_sum = pd.DataFrame(pivot_table.sum().round(0), columns=[''])
pivot_table = pd.concat([pivot_table, pd.DataFrame(column_sum.T)])

# station_totals = pivot_table.sum()
# station_totals.name = 'total'
# pivot_table = pivot_table.append(station_totals)

print(pivot_table)



fig, ax = plt.subplots(figsize=(15, 6))

months = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
          'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Total']
# 繪製表格
table = ax.table(cellText=pivot_table.values, colLabels=pivot_table.columns, rowLabels=months,
                 cellLoc='center', loc='center')

# 設定表格樣式
table.auto_set_font_size(False)
table.set_fontsize(12)
table.scale(1.2, 1.2)

# 隱藏座標軸
ax.axis('off')


plt.figure(figsize=(6, 6))
# column_sum = pd.DataFrame(pivot_table.sum(), columns=[''])
x = pd.DataFrame(df.drop(columns=['datetime']).sum(), columns=['']).to_numpy().reshape(-1)
plt.pie(x,
        radius=1.5,
        labels=col_name,
        autopct='%.1f%%')

# 顯示圖表
plt.show()