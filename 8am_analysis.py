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

df['datetime'] = df['date'] + ' ' + df['time']

df['datetime'] = pd.to_datetime(df['datetime'], format='%d.%m.%Y %H:%M:%S')
df['year'] = df['datetime'].dt.year
df['month'] = df['datetime'].dt.month
df['day'] = df['datetime'].dt.day
df.dropna(axis=0, inplace=True)

# df.drop(columns=['date', 'time'], inplace=True)

print(df.info())

selected_data = df[df['time'] == '08:00:00'].drop(columns=['time', 'datetime'])

cross_tab = pd.crosstab(selected_data['month'], selected_data['fleher deich west stromabwaerts'])

# chi2, p_value, _, _ = chi2_contingency(cross_tab, correction = True)
print(chi2_contingency(cross_tab, correction = False))

# 輸出卡方檢驗結果
# print(f"Chi-square value: {chi2}")
# print(f"P-value: {p_value}")