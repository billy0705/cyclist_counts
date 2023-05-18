import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv('cyclist_counts.csv', delimiter=';')
missing_rows = df[df.isnull().any(axis=1)]
print(missing_rows)

# selected_data = df[df['time'] == '02:00:00'].drop(columns=['date', 'time'])
# analysis the data in time == '02:00:00'
# max_values = selected_data.max()
# min_values = selected_data.min()
# mean_values = selected_data.mean()
# median_values = selected_data.median()
# quantile25_values = selected_data.quantile(0.25)
# quantile75_values = selected_data.quantile(0.75)
# quantile90_values = selected_data.quantile(0.90)
# kurt_values = selected_data.kurt()
# skew_values = selected_data.skew()

# print(max_values)
# print(min_values)
# print(mean_values)
# print(median_values)
# print(quantile25_values)
# print(quantile75_values)
# print(quantile90_values)
# print(kurt_values)
# print(skew_values)

time_select = ['02:00:00', '02:15:00', '02:30:00', '02:45:00']
for i, stime in enumerate(time_select):
    selected_data = df[df['time'] == stime].drop(columns=['date', 'time'])
    print(i)
    df_melted = pd.melt(selected_data[['fleher deich ost stromaufwaerts', 'fleher deich west stromabwaerts', 'okb nord', 'okb sued']], var_name='Columns', value_name='Counts')
    plt.figure(figsize=(15, 6))
    ax = sns.boxplot(x='Columns', y='Counts', data=df_melted)
    ax.set_xticklabels(ax.get_xticklabels(), wrap=True)
    title_name = "Data distribution in time = " + stime
    ax.set_title(title_name)
plt.show()


################# fill in missing row and change the wrong time
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