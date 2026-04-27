"""
=============================================================
  Adaptive ML Pipeline + Cloud Cost Optimizer
  IoT Migration Cost Prediction
  Author: Ritika | Auburn University at Montgomery

  USAGE:
    # Run everything in one command:
    python pipeline.py --data RT_IOT2022.csv

    # Skip optimizer (faster, ML only):
    python pipeline.py --data RT_IOT2022.csv --no-optimizer

    # Predict costs for new IoT flows:
    python pipeline.py --predict new_flows.csv

    # Watch a folder for new CSV files automatically:
    python pipeline.py --watch ./data_folder
=============================================================
"""

import argparse
import os
import sys
import time
import glob
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import joblib

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_absolute_error, mean_squared_error,
                             r2_score, explained_variance_score)
from xgboost import XGBRegressor

np.random.seed(42)

# =============================================================
#  CONFIGURATION
# =============================================================
FEATURE_COLS = [
    'flow_duration', 'fwd_pkts_tot', 'bwd_pkts_tot',
    'flow_pkts_per_sec', 'fwd_header_size_tot', 'bwd_header_size_tot',
    'flow_iat.avg', 'flow_iat.std', 'active.avg', 'idle.avg',
    'data_volume_gb', 'cloud_provider', 'storage_type', 'region',
]
TARGET_COL      = 'migration_cost_usd'
PROVIDER_RATE   = {'AWS': 0.09,  'Azure': 0.087, 'GCP': 0.08}
STORAGE_MULT    = {'Standard': 1.0, 'Infrequent': 0.6, 'Archive': 0.2}
REGION_MULT     = {'us-east-1': 1.0, 'us-west-2': 1.05,
                   'eu-west-1': 1.15, 'ap-southeast-1': 1.20}
REGIONS         = ['us-east-1', 'us-west-2', 'eu-west-1', 'ap-southeast-1']
DRIFT_THRESHOLD = 0.15

CLOUD_CONFIG = {
    'AWS': {
        'transfer_rate': 0.09, 'compute_rate': 0.08,
        'regions': {
            'us-east-1'     : {'cost_mult': 1.00, 'carbon_kg_per_kwh': 0.415},
            'us-west-2'     : {'cost_mult': 1.30, 'carbon_kg_per_kwh': 0.058},
            'eu-west-1'     : {'cost_mult': 1.50, 'carbon_kg_per_kwh': 0.316},
            'ap-southeast-1': {'cost_mult': 1.80, 'carbon_kg_per_kwh': 0.493},
        }
    },
    'Azure': {
        'transfer_rate': 0.07, 'compute_rate': 0.09,
        'regions': {
            'us-east-1'     : {'cost_mult': 1.05, 'carbon_kg_per_kwh': 0.385},
            'us-west-2'     : {'cost_mult': 1.35, 'carbon_kg_per_kwh': 0.044},
            'eu-west-1'     : {'cost_mult': 1.55, 'carbon_kg_per_kwh': 0.228},
            'ap-southeast-1': {'cost_mult': 1.85, 'carbon_kg_per_kwh': 0.471},
        }
    },
    'GCP': {
        'transfer_rate': 0.08, 'compute_rate': 0.07,
        'regions': {
            'us-east-1'     : {'cost_mult': 1.08, 'carbon_kg_per_kwh': 0.394},
            'us-west-2'     : {'cost_mult': 1.25, 'carbon_kg_per_kwh': 0.067},
            'eu-west-1'     : {'cost_mult': 1.48, 'carbon_kg_per_kwh': 0.241},
            'ap-southeast-1': {'cost_mult': 1.75, 'carbon_kg_per_kwh': 0.455},
        }
    }
}
STORAGE_CONFIG = {
    'Standard'  : {'cost_mult': 1.0},
    'Infrequent': {'cost_mult': 0.6},
    'Archive'   : {'cost_mult': 0.2},
}
CARBON_WEIGHT = 0.3
COST_WEIGHT   = 0.7
POWER_PER_GB  = 0.00006


# =============================================================
#  STEP 1 — Load Data
# =============================================================
def load_data(path):
    print(f'\n[1/7] Loading dataset: {path}')
    if not os.path.exists(path):
        raise FileNotFoundError(f'Dataset not found: {path}')
    df = pd.read_csv(path)
    print(f'      Rows    : {df.shape[0]:,}')
    print(f'      Columns : {df.shape[1]}')
    print(f'      Attack types: {df["Attack_type"].nunique()}')
    return df


# =============================================================
#  STEP 2 — Add Cloud & Cost Columns
# =============================================================
def assign_provider(t):
    if 'MQTT' in str(t) or 'Thing' in str(t) or 'Wipro' in str(t):
        return 'AWS'
    elif 'NMAP' in str(t) or 'Metasploit' in str(t):
        return 'Azure'
    else:
        return 'GCP'


def add_cost_columns(df):
    print('\n[2/7] Adding cloud and cost columns...')
    total_bytes        = (df['fwd_pkts_payload.tot'].fillna(0)
                        + df['bwd_pkts_payload.tot'].fillna(0))
    data_gb            = (total_bytes / (1024**3)).clip(lower=1e-6)
    df['cloud_provider'] = df['Attack_type'].apply(assign_provider)
    df['region']         = np.random.choice(REGIONS, size=len(df), p=[0.4,0.3,0.2,0.1])
    total_payload        = (df['fwd_pkts_payload.tot'].fillna(0)
                          + df['bwd_pkts_payload.tot'].fillna(0))
    df['storage_type']   = pd.cut(total_payload, bins=[-1,100,1000,float('inf')],
                                  labels=['Archive','Infrequent','Standard'])
    df['data_volume_gb'] = data_gb.round(8)

    median_rate   = df['payload_bytes_per_second'].median()
    peak_mult     = np.where(df['payload_bytes_per_second'] > median_rate, 1.3, 1.0)
    duration_hrs  = (df['flow_duration'] / 3600).clip(lower=0.001, upper=1.0)
    compute_cost  = duration_hrs * 50
    packet_cost   = (df['fwd_pkts_tot'] + df['bwd_pkts_tot']).clip(upper=10000) * 0.002
    header_cost   = (df['fwd_header_size_tot'] + df['bwd_header_size_tot']).clip(upper=100000) * 0.0001
    iat_cost      = df['flow_iat.avg'].clip(lower=0, upper=1000) * 0.01
    active_cost   = df['active.avg'].clip(lower=0, upper=1000) * 0.005
    transfer_cost = (
        df['data_volume_gb'] * 1000
        * df['cloud_provider'].map(PROVIDER_RATE)
        * df['storage_type'].astype(str).map(STORAGE_MULT)
        * df['region'].map(REGION_MULT)
    ).clip(upper=500)
    if transfer_cost.isna().sum() > 0:
        transfer_cost = transfer_cost.fillna(0)

    noise      = np.random.normal(0, 0.05, len(df))
    base_cost  = compute_cost + packet_cost + header_cost + iat_cost + active_cost + transfer_cost
    total_cost = (base_cost * peak_mult) * (1 + noise)
    df['migration_cost_usd'] = total_cost.clip(lower=0.01, upper=100).round(4)

    print(f'      Min: ${df["migration_cost_usd"].min():.4f}  '
          f'Max: ${df["migration_cost_usd"].max():.4f}  '
          f'Mean: ${df["migration_cost_usd"].mean():.4f}')
    return df


# =============================================================
#  STEP 3 — Feature Engineering
# =============================================================
def feature_engineering(df):
    print('\n[3/7] Feature engineering...')
    df_model = df[FEATURE_COLS + [TARGET_COL]].copy()

    num_cols = df_model.select_dtypes(include='number').columns
    df_model[num_cols] = df_model[num_cols].fillna(df_model[num_cols].median())
    cat_cols = df_model.select_dtypes(include='object').columns
    df_model[cat_cols] = df_model[cat_cols].fillna('Unknown')

    encoders  = {}
    text_cols = ['cloud_provider', 'storage_type', 'region']
    for col in text_cols:
        encoders[col] = LabelEncoder()
        df_model[col] = encoders[col].fit_transform(df_model[col].astype(str))

    df_model['cost_log'] = np.log1p(df_model[TARGET_COL])
    X        = df_model[FEATURE_COLS].values
    y        = df_model['cost_log'].values
    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42)

    print(f'      Train: {X_train.shape[0]:,}  Test: {X_test.shape[0]:,}  '
          f'Features: {X_train.shape[1]}')

    joblib.dump(scaler,   'scaler.pkl')
    joblib.dump(encoders, 'encoders.pkl')
    with open('feature_names.json', 'w') as f:
        json.dump(FEATURE_COLS, f)

    return df_model, X_scaled, X_train, X_test, y_train, y_test, scaler, encoders


# =============================================================
#  STEP 4 — Train Models
# =============================================================
def evaluate_model(name, model, X_tr, X_te, y_tr, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    mae   = mean_absolute_error(y_te, preds)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    r2    = r2_score(y_te, preds)
    evs   = explained_variance_score(y_te, preds)
    mask  = y_te != 0
    mape  = np.mean(np.abs((y_te[mask] - preds[mask]) / y_te[mask])) * 100
    print(f'      {name:25s}  MAE:{mae:.4f}  RMSE:{rmse:.4f}  '
          f'R²:{r2:.4f}  EVS:{evs:.4f}  MAPE:{mape:.2f}%')
    return {'Model': name, 'MAE': round(mae,4), 'RMSE': round(rmse,4),
            'R2': round(r2,4), 'EVS': round(evs,4), 'MAPE': round(mape,2),
            'predictions': preds, 'object': model}


def train_models(X_train, X_test, y_train, y_test):
    print('\n[4/7] Training models...')
    results = []
    for name, model in [
        ('Linear Regression', LinearRegression()),
        ('Decision Tree',     DecisionTreeRegressor(max_depth=10, random_state=42)),
        ('Random Forest',     RandomForestRegressor(n_estimators=100, max_depth=10,
                                                    random_state=42, n_jobs=-1)),
        ('XGBoost',           XGBRegressor(n_estimators=200, max_depth=6,
                                           learning_rate=0.1, random_state=42,
                                           verbosity=0)),
    ]:
        results.append(evaluate_model(name, model, X_train, X_test, y_train, y_test))

    best = min(results, key=lambda x: x['MAE'])
    print(f'\n      Best model: {best["Model"]}')
    joblib.dump(best['object'], 'best_model.pkl')
    with open('baseline_mae.json', 'w') as f:
        json.dump({'baseline_mae': best['MAE'], 'model': best['Model']}, f)
    print(f'      Saved: best_model.pkl  |  Baseline MAE: {best["MAE"]}')
    return results


# =============================================================
#  STEP 5 — Adaptive Retraining
# =============================================================
def adaptive_framework(X_scaled, y_full):
    print('\n[5/7] Running adaptive retraining framework...')
    n_windows, window_size, step_size = 10, 8000, 2000
    window_maes, window_rmses, window_r2s = [], [], []

    print(f'      {"Window":>8}  {"MAE":>8}  {"RMSE":>8}  {"R²":>8}')
    print(f'      {"-"*40}')

    for w in range(n_windows):
        start, end = w * step_size, w * step_size + window_size
        if end + 2000 > len(X_scaled):
            break
        model_w = XGBRegressor(n_estimators=100, max_depth=6,
                               learning_rate=0.1, random_state=42, verbosity=0)
        model_w.fit(X_scaled[start:end], y_full[start:end])
        preds_w = model_w.predict(X_scaled[end:end+2000])
        mae_w   = mean_absolute_error(y_full[end:end+2000], preds_w)
        rmse_w  = np.sqrt(mean_squared_error(y_full[end:end+2000], preds_w))
        r2_w    = r2_score(y_full[end:end+2000], preds_w)
        window_maes.append(mae_w)
        window_rmses.append(rmse_w)
        window_r2s.append(r2_w)
        print(f'      {w+1:>8}  {mae_w:>8.4f}  {rmse_w:>8.4f}  {r2_w:>8.4f}')

    improvement = ((window_maes[0] - window_maes[-1]) / window_maes[0]) * 100
    print(f'\n      Starting MAE: {window_maes[0]:.4f}  '
          f'Final MAE: {window_maes[-1]:.4f}  '
          f'Improvement: {improvement:.1f}%')
    return window_maes, window_rmses, window_r2s, improvement


# =============================================================
#  DRIFT DETECTION
# =============================================================
def check_drift(current_mae):
    if not os.path.exists('baseline_mae.json'):
        print('      No baseline found — skipping drift check.')
        return False
    with open('baseline_mae.json') as f:
        data = json.load(f)
    baseline_mae = data['baseline_mae']
    drift        = (current_mae - baseline_mae) / baseline_mae
    print(f'\n--- DRIFT DETECTION ---')
    print(f'      Baseline MAE : {baseline_mae:.4f}')
    print(f'      Current MAE  : {current_mae:.4f}')
    print(f'      Drift        : {drift*100:.1f}%')
    if drift > DRIFT_THRESHOLD:
        print(f'      STATUS: DRIFT DETECTED — triggering retraining!')
        return True
    print(f'      STATUS: Model stable — no retraining needed.')
    return False


# =============================================================
#  STEP 6 — Cloud Cost Optimizer (runs automatically)
# =============================================================
def calculate_cost_carbon(row, provider, region, storage):
    cfg          = CLOUD_CONFIG[provider]
    region_cfg   = cfg['regions'][region]
    storage_mult = STORAGE_CONFIG[storage]['cost_mult']
    cost_mult    = region_cfg['cost_mult']
    carbon_factor= region_cfg['carbon_kg_per_kwh']

    transfer_cost = row['data_volume_gb'] * 1000 * cfg['transfer_rate'] * storage_mult * cost_mult
    compute_cost  = (row['flow_duration'] / 3600) * cfg['compute_rate'] * cost_mult * 500
    packet_cost   = (row['fwd_pkts_tot'] + row['bwd_pkts_tot']) * 0.005 * cost_mult

    attack = str(row.get('Attack_type', ''))
    if provider == 'AWS' and ('MQTT' in attack or 'Thing' in attack or 'Wipro' in attack):
        compute_cost  *= 0.70
    elif provider == 'Azure':
        if 'Metasploit' in attack or 'NMAP' in attack or 'Brute' in attack:
            transfer_cost *= 0.60
        if region == 'eu-west-1':
            compute_cost  *= 0.68
        if region == 'ap-southeast-1':
            transfer_cost *= 0.65
    elif provider == 'GCP':
        packet_cost   *= 0.70
        compute_cost  *= 0.85
        if region == 'us-west-2':
            compute_cost  *= 0.65
        if region == 'ap-southeast-1':
            compute_cost  *= 0.70

    total_cost = max(0.01, min(100, transfer_cost + compute_cost + packet_cost))
    energy_kwh = row['data_volume_gb'] * 1000 * POWER_PER_GB
    carbon_kg  = energy_kwh * carbon_factor

    return round(total_cost, 4), round(carbon_kg, 6)


def run_optimizer(df):
    print('\n[6/7] Running cloud cost optimizer...')
    print(f'      Evaluating {len(df):,} flows x 3 providers x 4 regions x 3 storage'
          f' = {len(df)*36:,} combinations')

    results = []
    for idx, row in df.iterrows():
        all_options = []
        worst_cost  = 0

        for provider in CLOUD_CONFIG:
            for region in CLOUD_CONFIG[provider]['regions']:
                for storage in STORAGE_CONFIG:
                    cost, carbon = calculate_cost_carbon(row, provider, region, storage)
                    all_options.append({
                        'provider': provider, 'region': region,
                        'storage': storage, 'cost': cost, 'carbon': carbon
                    })
                    if cost > worst_cost:
                        worst_cost = cost

        max_cost   = max(o['cost']   for o in all_options)
        min_cost   = min(o['cost']   for o in all_options)
        max_carbon = max(o['carbon'] for o in all_options)
        min_carbon = min(o['carbon'] for o in all_options)

        for o in all_options:
            cost_score   = (o['cost']   - min_cost)   / (max_cost   - min_cost   + 1e-9)
            carbon_score = (o['carbon'] - min_carbon) / (max_carbon - min_carbon + 1e-9)
            o['combined_score'] = COST_WEIGHT * cost_score + CARBON_WEIGHT * carbon_score

        best  = min(all_options, key=lambda x: x['combined_score'])
        cheap = min(all_options, key=lambda x: x['cost'])
        green = min(all_options, key=lambda x: x['carbon'])

        savings     = round(worst_cost - best['cost'], 4)
        savings_pct = round((savings / worst_cost) * 100, 1) if worst_cost > 0 else 0

        results.append({
            'flow_id'              : idx,
            'attack_type'          : row['Attack_type'],
            'data_volume_gb'       : round(row['data_volume_gb'], 6),
            'recommended_provider' : best['provider'],
            'recommended_region'   : best['region'],
            'recommended_storage'  : best['storage'],
            'recommended_cost_usd' : best['cost'],
            'recommended_carbon_kg': best['carbon'],
            'cheapest_provider'    : cheap['provider'],
            'cheapest_cost_usd'    : cheap['cost'],
            'greenest_provider'    : green['provider'],
            'greenest_carbon_kg'   : green['carbon'],
            'worst_cost_usd'       : round(worst_cost, 4),
            'savings_usd'          : savings,
            'savings_pct'          : savings_pct,
        })

        if (idx + 1) % 10000 == 0:
            print(f'      Processed {idx+1:,} flows...')

    results_df = pd.DataFrame(results)
    results_df.to_csv('optimizer_recommendations.csv', index=False)
    avg_savings = results_df['savings_pct'].mean()
    total_saved = results_df['savings_usd'].sum()
    print(f'      Done! Avg savings: {avg_savings:.1f}%  Total saved: ${total_saved:,.2f}')
    print(f'      Saved: optimizer_recommendations.csv')
    return results_df


# =============================================================
#  STEP 7 — Save All 7 Figures
# =============================================================
def save_figures(df, df_model, results, y_test,
                 window_maes, window_rmses, window_r2s,
                 optimizer_df=None):
    print('\n[7/7] Saving figures...')

    # Figure 1 — EDA
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    fig.suptitle('RT-IoT2022 — Exploratory Data Analysis',
                 fontsize=16, fontweight='bold')
    axes[0,0].hist(np.log1p(df['migration_cost_usd']), bins=60,
                   color='steelblue', edgecolor='white')
    axes[0,0].set_title('Cost Distribution (log scale)')
    axes[0,0].set_xlabel('log(1 + Cost USD)')
    axes[0,0].set_ylabel('Frequency')
    for provider, color in {'AWS':'#FF9900','Azure':'#0078D4','GCP':'#4285F4'}.items():
        axes[0,1].hist(np.log1p(df[df['cloud_provider']==provider]['migration_cost_usd']),
                       bins=40, alpha=0.6, label=provider, color=color)
    axes[0,1].set_title('Cost by Cloud Provider')
    axes[0,1].legend()
    sample = df.sample(3000)
    axes[0,2].scatter(np.log1p(sample['data_volume_gb']),
                      np.log1p(sample['migration_cost_usd']),
                      alpha=0.3, color='coral', s=8)
    axes[0,2].set_title('Cost vs Data Volume')
    attack_counts = df['Attack_type'].value_counts()
    axes[1,0].barh(attack_counts.index, attack_counts.values,
                   color=plt.cm.Set3(np.linspace(0,1,len(attack_counts))))
    axes[1,0].set_title('IoT Traffic Types')
    provider_counts = df['cloud_provider'].value_counts()
    axes[1,1].bar(provider_counts.index, provider_counts.values,
                  color=['#FF9900','#0078D4','#4285F4'])
    axes[1,1].set_title('Samples per Cloud Provider')
    axes[1,2].hist(np.log1p(df['payload_bytes_per_second']), bins=50,
                   color='mediumpurple', edgecolor='white')
    axes[1,2].set_title('Payload Rate (log scale)')
    plt.tight_layout()
    plt.savefig('figure1_eda_plots.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved: figure1_eda_plots.png')

    # Figure 2 — Correlation
    key_cols = ['flow_duration','fwd_pkts_tot','bwd_pkts_tot',
                'payload_bytes_per_second','fwd_pkts_payload.tot',
                'bwd_pkts_payload.tot','data_volume_gb','migration_cost_usd']
    plt.figure(figsize=(10, 7))
    sns.heatmap(df[key_cols].corr(), annot=True, fmt='.2f', cmap='coolwarm',
                center=0, linewidths=0.5, annot_kws={'size':10})
    plt.title('Feature Correlation Heatmap', fontsize=13)
    plt.tight_layout()
    plt.savefig('figure2_correlation.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved: figure2_correlation.png')

    # Figure 3 — Log transform
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].hist(df_model[TARGET_COL], bins=60, color='steelblue', edgecolor='white')
    axes[0].set_title('Cost — BEFORE log transform')
    axes[0].set_xlabel('Migration Cost (USD)')
    axes[1].hist(df_model['cost_log'], bins=60, color='teal', edgecolor='white')
    axes[1].set_title('Cost — AFTER log transform')
    axes[1].set_xlabel('log(1 + Cost USD)')
    plt.tight_layout()
    plt.savefig('figure3_log_transform.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved: figure3_log_transform.png')

    # Figure 4 — Actual vs Predicted
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Actual vs Predicted — All Models',
                 fontsize=15, fontweight='bold')
    for i, r in enumerate(results):
        ax = axes.flatten()[i]
        ax.scatter(y_test, r['predictions'], alpha=0.3, s=8,
                   color=['steelblue','coral','teal','darkorchid'][i])
        mn = min(y_test.min(), r['predictions'].min())
        mx = max(y_test.max(), r['predictions'].max())
        ax.plot([mn,mx],[mn,mx],'r--',linewidth=1.5,label='Perfect')
        ax.set_title(f"{r['Model']}\nMAE={r['MAE']}  R²={r['R2']}  MAPE={r['MAPE']}%")
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.legend(fontsize=9)
    plt.tight_layout()
    plt.savefig('figure4_actual_vs_predicted.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved: figure4_actual_vs_predicted.png')

    # Figure 5 — Feature Importance
    xgb_model = next(r['object'] for r in results if r['Model'] == 'XGBoost')
    feat_imp  = pd.Series(xgb_model.feature_importances_,
                          index=FEATURE_COLS).sort_values(ascending=True)
    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp.index, feat_imp.values, color='darkorchid', edgecolor='white')
    plt.xlabel('Feature Importance Score')
    plt.title('XGBoost — Feature Importance', fontsize=13)
    plt.tight_layout()
    plt.savefig('figure5_feature_importance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved: figure5_feature_importance.png')

    # Figure 6 — Adaptive performance
    window_nums = list(range(1, len(window_maes)+1))
    fig, axes   = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle('Adaptive Retraining — Performance Over Time',
                 fontsize=14, fontweight='bold')
    for ax, vals, label, color, marker in zip(
        axes,
        [window_maes, window_rmses, window_r2s],
        ['MAE\n(lower=better)', 'RMSE\n(lower=better)', 'R²\n(higher=better)'],
        ['steelblue', 'coral', 'teal'], ['o', 's', '^']
    ):
        ax.plot(window_nums, vals, marker=marker, color=color, linewidth=2, markersize=6)
        ax.fill_between(window_nums, vals, alpha=0.1, color=color)
        ax.set_title(label)
        ax.set_xlabel('Retraining Window')
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figure6_adaptive_performance.png', dpi=150, bbox_inches='tight')
    plt.close()
    print('      Saved: figure6_adaptive_performance.png')

    # Figure 7 — Optimizer (only if optimizer ran)
    if optimizer_df is not None:
        colors = {'AWS': '#FF9900', 'Azure': '#0078D4', 'GCP': '#4285F4'}
        fig, axes = plt.subplots(2, 3, figsize=(18, 10))
        fig.suptitle('Cloud Cost Optimizer — Analysis Report',
                     fontsize=16, fontweight='bold')

        provider_counts = optimizer_df['recommended_provider'].value_counts()
        axes[0,0].bar(provider_counts.index, provider_counts.values,
                      color=[colors[p] for p in provider_counts.index])
        axes[0,0].set_title('Recommended Cloud Provider\n(Cost + Carbon Optimized)')
        axes[0,0].set_xlabel('Provider')
        axes[0,0].set_ylabel('Number of IoT Flows')
        for i, (p, v) in enumerate(provider_counts.items()):
            axes[0,0].text(i, v+100, f'{v:,}', ha='center', fontsize=10, fontweight='bold')

        axes[0,1].hist(optimizer_df['savings_pct'], bins=50, color='teal', edgecolor='white')
        axes[0,1].set_title('Cost Savings Distribution\n(vs Worst Option)')
        axes[0,1].set_xlabel('Savings (%)')
        axes[0,1].axvline(optimizer_df['savings_pct'].mean(), color='red',
                          linestyle='--', linewidth=2,
                          label=f'Mean: {optimizer_df["savings_pct"].mean():.1f}%')
        axes[0,1].legend()

        sample = optimizer_df.sample(min(500, len(optimizer_df)))
        axes[0,2].scatter(sample['worst_cost_usd'], sample['recommended_cost_usd'],
                          alpha=0.4, color='coral', s=10)
        mn = min(sample['worst_cost_usd'].min(), sample['recommended_cost_usd'].min())
        mx = max(sample['worst_cost_usd'].max(), sample['recommended_cost_usd'].max())
        axes[0,2].plot([mn,mx],[mn,mx],'r--',linewidth=1.5,label='No savings line')
        axes[0,2].set_title('Optimized vs Worst Cost')
        axes[0,2].set_xlabel('Worst Cost (USD)')
        axes[0,2].set_ylabel('Recommended Cost (USD)')
        axes[0,2].legend()

        carbon_by_provider = optimizer_df.groupby(
            'recommended_provider')['recommended_carbon_kg'].mean()
        axes[1,0].bar(carbon_by_provider.index, carbon_by_provider.values,
                      color=[colors[p] for p in carbon_by_provider.index])
        axes[1,0].set_title('Avg Carbon Footprint by Provider (kg CO2)')
        axes[1,0].set_xlabel('Provider')

        region_counts = optimizer_df['recommended_region'].value_counts()
        axes[1,1].barh(region_counts.index, region_counts.values, color='steelblue')
        axes[1,1].set_title('Recommended Region Distribution')
        axes[1,1].set_xlabel('Number of Flows')

        sample2        = optimizer_df.sample(min(1000, len(optimizer_df)))
        scatter_colors = [colors[p] for p in sample2['recommended_provider']]
        axes[1,2].scatter(sample2['recommended_cost_usd'],
                          sample2['recommended_carbon_kg'],
                          c=scatter_colors, alpha=0.5, s=10)
        axes[1,2].set_title('Cost vs Carbon Tradeoff')
        axes[1,2].set_xlabel('Recommended Cost (USD)')
        axes[1,2].set_ylabel('Carbon Footprint (kg CO2)')
        patches = [mpatches.Patch(color=colors[p], label=p) for p in colors]
        axes[1,2].legend(handles=patches)

        plt.tight_layout()
        plt.savefig('figure7_optimizer_analysis.png', dpi=150, bbox_inches='tight')
        plt.close()
        print('      Saved: figure7_optimizer_analysis.png')


# =============================================================
#  PREDICTION MODE
# =============================================================
def predict_new_data(input_path):
    print(f'\n--- PREDICTING COSTS FOR: {input_path} ---')
    if not os.path.exists('best_model.pkl'):
        print('ERROR: No trained model found. Run pipeline first:')
        print('  python pipeline.py --data RT_IOT2022.csv')
        return
    model    = joblib.load('best_model.pkl')
    scaler   = joblib.load('scaler.pkl')
    encoders = joblib.load('encoders.pkl')
    df_new   = pd.read_csv(input_path)
    print(f'      Loaded {len(df_new):,} rows')
    if 'cloud_provider' not in df_new.columns:
        df_new = add_cost_columns(df_new)
    for col in ['cloud_provider', 'storage_type', 'region']:
        if col in encoders:
            df_new[col] = df_new[col].map(
                lambda x: encoders[col].transform([str(x)])[0]
                if str(x) in encoders[col].classes_ else 0)
    for col in FEATURE_COLS:
        if col not in df_new.columns:
            df_new[col] = 0
    preds = np.expm1(model.predict(
        scaler.transform(df_new[FEATURE_COLS].fillna(0).values)))
    df_new['predicted_cost_usd'] = preds.round(4)
    output_path = input_path.replace('.csv', '_predictions.csv')
    df_new[['predicted_cost_usd']].to_csv(output_path, index=False)
    print(f'      Min: ${preds.min():.4f}  Max: ${preds.max():.4f}  '
          f'Mean: ${preds.mean():.4f}')
    print(f'      Predictions saved: {output_path}')


# =============================================================
#  WATCH FOLDER MODE
# =============================================================
def watch_folder(folder_path, interval=30):
    print(f'\n--- REAL-TIME MONITORING MODE ---')
    print(f'      Watching : {folder_path}')
    print(f'      Interval : every {interval} seconds')
    print(f'      Press Ctrl+C to stop\n')
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)
    processed = set(glob.glob(f'{folder_path}/*.csv'))
    print(f'      Skipping {len(processed)} existing files. Waiting...\n')
    try:
        while True:
            new_files = set(glob.glob(f'{folder_path}/*.csv')) - processed
            if new_files:
                for f in sorted(new_files):
                    print(f'\n NEW FILE: {f}  [{time.strftime("%H:%M:%S")}]')
                    try:
                        df       = load_data(f)
                        df       = add_cost_columns(df)
                        df_model, X_scaled, X_train, X_test, \
                            y_train, y_test, scaler, encoders = feature_engineering(df)
                        needs_retrain = True
                        if os.path.exists('best_model.pkl'):
                            preds_check   = joblib.load('best_model.pkl').predict(X_test)
                            needs_retrain = check_drift(
                                mean_absolute_error(y_test, preds_check))
                        if needs_retrain:
                            results      = train_models(X_train, X_test, y_train, y_test)
                            wm, wr, wr2, imp = adaptive_framework(
                                X_scaled, df_model['cost_log'].values)
                            optimizer_df = run_optimizer(df)
                            save_figures(df, df_model, results, y_test,
                                         wm, wr, wr2, optimizer_df)
                            print_summary(results, wm, imp, optimizer_df)
                        else:
                            print('      Model stable — skipping retraining.')
                        processed.add(f)
                    except Exception as e:
                        print(f'      ERROR: {e}')
                        processed.add(f)
            else:
                print(f'      [{time.strftime("%H:%M:%S")}] Waiting...', end='\r')
            time.sleep(interval)
    except KeyboardInterrupt:
        print('\n\nMonitoring stopped.')


# =============================================================
#  FINAL SUMMARY
# =============================================================
def print_summary(results, window_maes, improvement, optimizer_df=None):
    print()
    print('=' * 65)
    print('  PROJECT COMPLETE — FINAL SUMMARY')
    print('=' * 65)
    print('\nDATASET')
    print('  Name     : RT-IoT2022 (real IoT network traffic)')
    print('  Features : 14 selected from 85')
    print('  Target   : migration_cost_usd (synthetic label)')
    print('\nMODEL RESULTS')
    print(f'  {"Model":25s}  {"MAE":>7}  {"RMSE":>7}  {"R²":>7}  {"EVS":>7}  {"MAPE%":>7}')
    print(f'  {"-"*65}')
    for r in results:
        print(f"  {r['Model']:25s}  {r['MAE']:>7}  {r['RMSE']:>7}  "
              f"{r['R2']:>7}  {r['EVS']:>7}  {r['MAPE']:>6}%")
    print('\nADAPTIVE FRAMEWORK')
    print(f'  Windows     : {len(window_maes)}')
    print(f'  Starting MAE: {window_maes[0]:.4f}')
    print(f'  Final MAE   : {window_maes[-1]:.4f}')
    print(f'  Improvement : {improvement:.1f}%')
    if optimizer_df is not None:
        print('\nCLOUD COST OPTIMIZER')
        print(f'  Flows analyzed : {len(optimizer_df):,}')
        print(f'  Avg savings    : {optimizer_df["savings_pct"].mean():.1f}% per flow')
        print(f'  Total saved    : ${optimizer_df["savings_usd"].sum():,.2f}')
        top = optimizer_df['recommended_provider'].value_counts().index[0]
        print(f'  Top provider   : {top}')
    print('\nFILES SAVED')
    print('  figure1_eda_plots.png            -> Figure 1')
    print('  figure2_correlation.png          -> Figure 2')
    print('  figure3_log_transform.png        -> Figure 3')
    print('  figure4_actual_vs_predicted.png  -> Figure 4')
    print('  figure5_feature_importance.png   -> Figure 5')
    print('  figure6_adaptive_performance.png -> Figure 6')
    if optimizer_df is not None:
        print('  figure7_optimizer_analysis.png   -> Figure 7')
        print('  optimizer_recommendations.csv    -> Full optimizer results')
    print('  best_model.pkl | scaler.pkl | encoders.pkl | baseline_mae.json')
    print()


# =============================================================
#  MAIN
# =============================================================
def main():
    parser = argparse.ArgumentParser(
        description='IoT Migration Cost — Adaptive ML Pipeline + Cloud Optimizer')
    parser.add_argument('--data',         type=str,
                        help='Train on this CSV dataset')
    parser.add_argument('--predict',      type=str,
                        help='Predict costs for this CSV file')
    parser.add_argument('--watch',        type=str,
                        help='Watch folder for new CSV files automatically')
    parser.add_argument('--no-optimizer', action='store_true',
                        help='Skip cloud optimizer (faster run, ML only)')
    args = parser.parse_args()

    if args.predict:
        predict_new_data(args.predict)
        return

    if args.watch:
        watch_folder(args.watch)
        return

    if not args.data:
        print('ERROR: Please provide a dataset. Example:')
        print('  python pipeline.py --data RT_IOT2022.csv')
        sys.exit(1)

    print('=' * 65)
    print('  IoT Migration Cost Prediction — Full Pipeline')
    print('=' * 65)
    print(f'  Dataset   : {args.data}')
    print(f'  Optimizer : {"disabled (--no-optimizer)" if args.no_optimizer else "enabled"}')
    print('=' * 65)

    df                                                       = load_data(args.data)
    df                                                       = add_cost_columns(df)
    df_model, X_scaled, X_train, X_test, y_train, y_test, \
        scaler, encoders                                     = feature_engineering(df)
    results                                                  = train_models(
                                                                X_train, X_test,
                                                                y_train, y_test)
    wm, wr, wr2, imp                                         = adaptive_framework(
                                                                X_scaled,
                                                                df_model['cost_log'].values)
    optimizer_df = None
    if not args.no_optimizer:
        optimizer_df = run_optimizer(df)

    save_figures(df, df_model, results, y_test, wm, wr, wr2, optimizer_df)
    print_summary(results, wm, imp, optimizer_df)


if __name__ == '__main__':
    main()
