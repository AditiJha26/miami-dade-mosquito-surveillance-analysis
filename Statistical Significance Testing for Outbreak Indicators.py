import pandas as pd
import numpy as np
from scipy.stats import mannwhitneyu, ttest_ind
import seaborn as sns
import matplotlib.pyplot as plt

# load data
file_path = r"C:/user/data/Miami_and_Hialeah_alltraps.xlsx"
df = pd.read_excel(file_path)

df['Date_Submitted'] = pd.to_datetime(df['Date_Submitted'])
df = df.sort_values('Date_Submitted')

def process_city_data(data):
    # group by week
    weekly = data.groupby(pd.Grouper(key='Date_Submitted', freq='W-MON'))['Cx_quinquefasciatus_fem'].mean().reset_index()
    weekly = weekly.dropna()
    
    # lag features (1-4 weeks)
    weekly['Lag 1'] = weekly['Cx_quinquefasciatus_fem'].shift(1)
    weekly['Lag 2'] = weekly['Cx_quinquefasciatus_fem'].shift(2)
    weekly['Lag 3'] = weekly['Cx_quinquefasciatus_fem'].shift(3)
    weekly['Lag 4'] = weekly['Cx_quinquefasciatus_fem'].shift(4)
    
    # rolling means
    weekly['Rolling Mean 2'] = weekly['Cx_quinquefasciatus_fem'].rolling(2).mean()
    weekly['Rolling Mean 4'] = weekly['Cx_quinquefasciatus_fem'].rolling(4).mean()
    
    # growth rate
    weekly['Growth Rate'] = (weekly['Cx_quinquefasciatus_fem'] - weekly['Lag 1']) / (weekly['Lag 1'])
    
    threshold = weekly['Cx_quinquefasciatus_fem'].quantile(0.80)
    weekly['Outbreak'] = weekly['Cx_quinquefasciatus_fem'] >= threshold
    return weekly.dropna()

datasets = {
    'Miami': process_city_data(df[df['City'] == 'Miami']),
    'Hialeah': process_city_data(df[df['City'] == 'Hialeah']),
    'Full Dataset': process_city_data(df)
}

features = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 4', 'Rolling Mean 2', 'Rolling Mean 4', 'Growth Rate']
results = []

for db_name, data in datasets.items():
    outbreak = data[data['Outbreak']]
    normal = data[~data['Outbreak']]
    
    for feat in features:
        # Mann-Whitney U test
        u_stat, u_p = mannwhitneyu(outbreak[feat], normal[feat], alternative='two-sided')
        
        # Welch's t-test
        t_stat, t_p = ttest_ind(outbreak[feat], normal[feat], equal_var=False, nan_policy='omit')
        
        results.append({
            'Dataset': db_name,
            'Feature': feat,
            'Outbreak Mean': round(outbreak[feat].mean(), 2),
            'Non-Outbreak Mean': round(normal[feat].mean(), 2),
            'Mann-Whitney p-value': u_p,
            'Welch p-value': t_p
        })

results_df = pd.DataFrame(results)

# results to csv
results_df.to_csv(r'C:/user/results', index=False)

# results to console
for db_name in ['Miami', 'Hialeah', 'Full Dataset']:
    print(f"\n==================== {db_name} Dataset ====================")
    display_df = results_df[results_df['Dataset'] == db_name].drop(columns=['Dataset'])
    print(display_df.to_string(index=False))

# p-value heatmap
heatmap_data = results_df.pivot(index='Feature', columns='Dataset', values='Mann-Whitney p-value')
heatmap_data = heatmap_data.reindex(features)

# plot settings
plt.figure(figsize=(9, 6))
sns.heatmap(
    heatmap_data, 
    annot=True, 
    cmap='coolwarm_r',
    fmt=".2e", 
    cbar_kws={'label': 'Mann-Whitney p-value'},
    linewidths=0.5
)

plt.title('Mann-Whitney p-values: Outbreak vs Non-Outbreak Weeks', pad=15)
plt.ylabel('Features', labelpad=10)
plt.xlabel('Datasets', labelpad=10)
plt.tight_layout()

plt.savefig(r'C:/user/visualizations', dpi=300)
plt.show()