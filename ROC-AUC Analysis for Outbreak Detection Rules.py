import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import roc_auc_score, roc_curve, precision_score, recall_score, f1_score

# load data
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
    
    # ground truth outbreak
    threshold = weekly['Cx_quinquefasciatus_fem'].quantile(0.80)
    weekly['Outbreak'] = (weekly['Cx_quinquefasciatus_fem'] >= threshold).astype(int)
    
    # high thresholds for rules
    weekly['Lag 1 High'] = (weekly['Lag 1'] >= weekly['Lag 1'].quantile(0.80)).astype(int)
    weekly['Lag 2 High'] = (weekly['Lag 2'] >= weekly['Lag 2'].quantile(0.80)).astype(int)
    weekly['Rolling Mean 2 High'] = (weekly['Rolling Mean 2'] >= weekly['Rolling Mean 2'].quantile(0.80)).astype(int)
    weekly['Rolling Mean 4 High'] = (weekly['Rolling Mean 4'] >= weekly['Rolling Mean 4'].quantile(0.80)).astype(int)
    
    # composite rules
    weekly['Rule: Lag 1 AND RM 4'] = (weekly['Lag 1 High'] & weekly['Rolling Mean 4 High']).astype(int)
    weekly['Rule: RM 2 AND RM 4'] = (weekly['Rolling Mean 2 High'] & weekly['Rolling Mean 4 High']).astype(int)
    
    return weekly

datasets = {
    'Miami': process_city_data(df[df['City'] == 'Miami']),
    'Hialeah': process_city_data(df[df['City'] == 'Hialeah']),
    'Full Dataset': process_city_data(df)
}

continuous_features = ['Lag 1', 'Lag 2', 'Lag 3', 'Lag 4', 'Rolling Mean 2', 'Rolling Mean 4', 'Growth Rate']
decision_rules = ['Lag 1 High', 'Lag 2 High', 'Rolling Mean 2 High', 'Rolling Mean 4 High', 'Rule: Lag 1 AND RM 4', 'Rule: RM 2 AND RM 4']

results = []
roc_curves_data = {}

for db_name, data in datasets.items():
    y_true = data['Outbreak']
    roc_curves_data[db_name] = {'continuous': {}, 'rules': {}}
    
    # continuous features
    for feat in continuous_features:
        auc = roc_auc_score(y_true, data[feat])
        fpr, tpr, _ = roc_curve(y_true, data[feat])
        roc_curves_data[db_name]['continuous'][feat] = (fpr, tpr, auc)
        
        results.append({
            'Dataset': db_name, 'Predictor': feat, 'Type': 'Continuous Feature',
            'Precision': np.nan, 'Recall': np.nan, 'F1': np.nan, 'ROC-AUC': auc
        })
        
    # binary decision rules
    for rule in decision_rules:
        y_pred = data[rule]
        
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        auc = roc_auc_score(y_true, y_pred)
        
        fpr, tpr, _ = roc_curve(y_true, y_pred)
        roc_curves_data[db_name]['rules'][rule] = (fpr, tpr, auc)
        
        results.append({
            'Dataset': db_name, 'Predictor': rule, 'Type': 'Decision Rule',
            'Precision': prec, 'Recall': rec, 'F1': f1, 'ROC-AUC': auc
        })
results_df = pd.DataFrame(results).round(4)

# sresults to csv
results_df.to_csv(r'C:/user/results', index=False)

# results to console
for db_name in ['Miami', 'Hialeah', 'Full Dataset']:
    print(f"\n==================== {db_name} Dataset ====================")
    display_df = results_df[results_df['Dataset'] == db_name].drop(columns=['Dataset'])
    print(display_df.to_string(index=False))

sns.set_theme(style="whitegrid")

for db_name in ['Miami', 'Hialeah', 'Full Dataset']:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    fig.suptitle(f'Outbreak Detection Performance: {db_name} Dataset', fontsize=16, weight='bold')
    
    # roc curves - continuous features
    ax = axes[0]
    for feat in ['Lag 1', 'Rolling Mean 2', 'Rolling Mean 4', 'Growth Rate']: 
        fpr, tpr, auc = roc_curves_data[db_name]['continuous'][feat]
        ax.plot(fpr, tpr, label=f'{feat} (AUC={auc:.2f})', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title('ROC: Individual Predictors')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    
    # roc curves - decision rules
    ax = axes[1]
    for rule in ['Lag 1 High', 'Rolling Mean 2 High', 'Rule: Lag 1 AND RM 4']:
        fpr, tpr, auc = roc_curves_data[db_name]['rules'][rule]
        ax.plot(fpr, tpr, marker='o', label=f'{rule} (AUC={auc:.2f})', lw=2)
    ax.plot([0, 1], [0, 1], 'k--', lw=1)
    ax.set_title('ROC: Decision Rules')
    ax.set_xlabel('False Positive Rate')
    ax.legend(loc='lower right')
    
    # comparison chart of auc
    ax = axes[2]
    subset = results_df[results_df['Dataset'] == db_name].sort_values('ROC-AUC', ascending=True)
    colors = ['skyblue' if t == 'Continuous Feature' else 'salmon' for t in subset['Type']]
    ax.barh(subset['Predictor'], subset['ROC-AUC'], color=colors, edgecolor='black')
    ax.set_title('ROC-AUC Comparison')
    ax.set_xlabel('ROC-AUC Score')
    ax.set_xlim(0.4, 1.0)
    
    legend_elements = [Patch(facecolor='skyblue', edgecolor='black', label='Continuous Feature'),
                       Patch(facecolor='salmon', edgecolor='black', label='Binary Rule')]
    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    file_name = f'roc_analysis_{db_name.replace(" ", "_").lower()}.png'
    plt.savefig(rf'file path to save visualizations\{file_name}', dpi=300)

plt.show()