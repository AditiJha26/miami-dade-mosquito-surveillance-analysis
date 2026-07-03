import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample

# load Data
file_path = r"C:/user/data/Miami_and_Hialeah_alltraps.xlsx"
df = pd.read_excel(file_path)

df['Date_Submitted'] = pd.to_datetime(df['Date_Submitted'])
df = df.sort_values('Date_Submitted')

def process_city_data(data):
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
    weekly['Growth Rate'] = (weekly['Cx_quinquefasciatus_fem'] - weekly['Lag 1']) / (weekly['Lag 1'] + 1e-5)
    
    weekly = weekly.dropna()
    
    #ground truth outbreak
    threshold = weekly['Cx_quinquefasciatus_fem'].quantile(0.80)
    weekly['Outbreak'] = (weekly['Cx_quinquefasciatus_fem'] >= threshold).astype(int)
    
    return weekly

datasets = {
    'Miami': process_city_data(df[df['City'] == 'Miami']),
    'Hialeah': process_city_data(df[df['City'] == 'Hialeah']),
    'Full Dataset': process_city_data(df)
}

features = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 4', 'Rolling Mean 2', 'Rolling Mean 4', 'Growth Rate']

# model fitting & bootstrapping
results = []
plot_data = {}

for db_name, data in datasets.items():
    X = data[features]
    y = data['Outbreak']
    
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_scaled = pd.DataFrame(X_scaled, columns=features)
    
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_scaled, y)
    
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    abs_importance = np.abs(coefs)

    n_bootstraps = 100
    boot_coefs = np.zeros((n_bootstraps, len(features)))
    
    for i in range(n_bootstraps):
        X_boot, y_boot = resample(X_scaled, y, random_state=i)
        boot_model = LogisticRegression(max_iter=1000)
        boot_model.fit(X_boot, y_boot)
        boot_coefs[i, :] = boot_model.coef_[0]
        
    coef_err = np.std(boot_coefs, axis=0)
    
    df_plot = pd.DataFrame({
        'Feature': features,
        'Coefficient': coefs,
        'Std_Error': coef_err,
        'Odds Ratio': odds_ratios,
        'Absolute Importance': abs_importance
    })
    
    df_plot = df_plot.sort_values(by='Absolute Importance', ascending=True).reset_index(drop=True)
    plot_data[db_name] = df_plot
    
    for _, row in df_plot.iterrows():
        results.append({
            'Dataset': db_name,
            'Feature': row['Feature'],
            'Coefficient': row['Coefficient'],
            'Std_Error': row['Std_Error'],
            'Odds Ratio': row['Odds Ratio'],
            'Absolute Importance': row['Absolute Importance']
        })

# report results
results_df = pd.DataFrame(results).round(4)

# results to csv
results_df.to_csv(r'C:/user/results', index=False)

# results to console
for db_name in ['Miami', 'Hialeah', 'Full Dataset']:
    print(f"\n==================== {db_name} Dataset ====================")
    display_df = results_df[results_df['Dataset'] == db_name].drop(columns=['Dataset'])
    print(display_df.sort_values(by='Absolute Importance', ascending=False).to_string(index=False))

sns.set_theme(style="whitegrid")

for db_name, df_plot in plot_data.items():
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Logistic Regression Feature Importance: {db_name}', fontsize=16, weight='bold')
    
    y_pos = np.arange(len(df_plot))
    
    # coefficients with bootstrap error bars
    ax = axes[0]
    ax.barh(y_pos, df_plot['Coefficient'], xerr=df_plot['Std_Error'], capsize=4, color='skyblue', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Feature'])
    ax.axvline(0, color='black', linewidth=1, linestyle='--')
    ax.set_title('Standardized Coefficients\n(with Bootstrap 95% CI)')
    ax.set_xlabel('Coefficient Value')
    
    # odds ratios
    ax = axes[1]
    colors = ['lightcoral' if val < 1 else 'lightgreen' for val in df_plot['Odds Ratio']]
    ax.barh(y_pos, df_plot['Odds Ratio'], color=colors, edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Feature'])
    ax.axvline(1, color='black', linewidth=1, linestyle='--') 
    ax.set_title('Odds Ratios')
    ax.set_xlabel('Odds Ratio (exp(coef))')
    
    # ranked absolute importance
    ax = axes[2]
    ax.barh(y_pos, df_plot['Absolute Importance'], color='mediumpurple', edgecolor='black')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_plot['Feature'])
    ax.set_title('Ranked Feature Importance\n(|Coefficient|)')
    ax.set_xlabel('Absolute Importance')

    plt.tight_layout()
    file_name = f'feature_importance_{db_name.replace(" ", "_").lower()}.png'
    plt.savefig(rf'file path to save visualizations\{file_name}', dpi=300)

plt.show()