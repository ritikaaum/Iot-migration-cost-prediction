# Adaptive ML Framework for IoT Migration Cost Prediction

**Author:** Ritika | Auburn University at Montgomery  
**Course:** Final Project | April 2026  
**GitHub:** https://github.com/ritikaaum/iot-migration-cost-prediction

---

## Project Overview

This project presents an Adaptive Machine Learning Framework for predicting and optimizing IoT cloud migration costs using the RT-IoT2022 real IoT network traffic dataset. The framework combines machine learning, adaptive retraining, SHAP explainability, and a multi-cloud cost optimizer to help organizations make data-driven cloud migration decisions.

---

## Key Results

| Metric | Value |
|--------|-------|
| Best Model | Random Forest |
| R² Score | 0.9996 |
| MAPE | 3.45% |
| MAE Improvement (Adaptive) | 88.4% |
| Average Cost Savings | 42.9% per flow |
| Total Savings Identified | $7,984.50 |
| Flows Analyzed | 123,117 |
| Combinations Evaluated | 4,432,212 |

---

## Statistical Validation

| Comparison | p-value | Result |
|------------|---------|--------|
| Random Forest vs Linear Regression | ~0.000000 | Significantly better |
| Random Forest vs XGBoost | 0.001060 | Significantly better |
| Random Forest vs Decision Tree | 0.633225 | Similar performance |

---

## Dataset

**RT-IoT2022** — Real IoT network traffic dataset  
- 123,117 rows, 85 features  
- 12 IoT traffic types (MQTT, NMAP, DOS, Wipro, Metasploit, etc.)  
- Available at: https://www.kaggle.com/datasets/dhoogla/rtiot2022  

**Important:** Download the dataset from Kaggle and place `RT_IOT2022.csv` in the same folder as `pipeline.py` before running.

---

## Installation

Install all required libraries with one command:

```bash
pip install numpy pandas matplotlib seaborn scikit-learn xgboost joblib shap scipy
```

---

## How to Run

### Full Pipeline — One Command Does Everything
```bash
python pipeline.py --data RT_IOT2022.csv
```

This single command automatically runs all 7 steps:
1. Loads the RT-IoT2022 dataset
2. Adds cloud and cost columns (AWS/Azure/GCP pricing)
3. Engineers 14 features from 85 original columns
4. Trains and compares 4 ML models
5. Runs adaptive retraining framework (10 windows)
6. Runs cloud cost optimizer (4.4M combinations)
7. Saves all 7 figures and results

### Predict Costs for New IoT Flows
```bash
python pipeline.py --predict new_flows.csv
```

### Watch Folder for Automatic Retraining
```bash
python pipeline.py --watch ./data_folder
```
Drop any new CSV file into the watched folder — pipeline automatically detects it, checks for model drift, and retrains if needed.

### Skip Optimizer (Faster Run — ML Only)
```bash
python pipeline.py --data RT_IOT2022.csv --no-optimizer
```

---
## Project Structure

---

## Models Compared

| Model | MAE | RMSE | R² | MAPE |
|-------|-----|------|----|------|
| Linear Regression | 0.2511 | 0.4580 | 0.757 | 183.33% |
| Decision Tree | 0.0091 | 0.0193 | 0.9996 | 3.45% |
| **Random Forest ★** | **0.0090** | **0.0190** | **0.9996** | **3.45%** |
| XGBoost | 0.0093 | 0.0200 | 0.9995 | 3.49% |

---

## Cloud Optimizer Results

| Provider | Flows Recommended | Use Case |
|----------|------------------|----------|
| GCP | 87.6% | General IoT traffic |
| AWS | 12.4% | MQTT/IoT Core traffic |
| Azure | — | Enterprise security (limited in this dataset) |

**Optimization formula:** 70% financial cost + 30% carbon footprint

---

## Output Files Generated After Running

After running `python pipeline.py --data RT_IOT2022.csv` the following files are automatically created:

- `figure1_eda_plots.png` — EDA analysis
- `figure2_correlation.png` — Feature correlations
- `figure3_log_transform.png` — Cost distribution transformation
- `figure4_actual_vs_predicted.png` — Model prediction quality
- `figure5_feature_importance.png` — Feature importance
- `figure6_adaptive_performance.png` — Adaptive retraining results
- `figure7_optimizer_analysis.png` — Optimizer analysis
- `figure8_shap_explanation.png` — SHAP feature explanations
- `optimizer_recommendations.csv` — Full recommendations for all flows
- `best_model.pkl` — Saved best model
- `scaler.pkl` — Fitted feature scaler
- `encoders.pkl` — Label encoders
- `baseline_mae.json` — Drift detection baseline

---

## Future Work

- Integrate real cloud billing data via AWS Cost Explorer API
- Validate on CIC-IoT2023 and UNSW-NB15 datasets
- Implement online learning for true real-time adaptation
- Add spot instance and reserved capacity pricing to optimizer
- Deploy as REST API for real-time IoT gateway integration
- Add reinforcement learning for dynamic pricing optimization

---

## References

1. RT-IoT2022 Dataset — https://www.kaggle.com/datasets/dhoogla/rtiot2022
2. Rajesh & Baghela (2025) — ML in Optimizing Cloud-Based Data Migration
3. Al-Masri et al. (2025) — Efficient Model to Estimate Cloud Migration Costs, Springer
4. Cui et al. (2024) — Multi-Objective VM Migration, Springer
5. Masud et al. (2021) — IoT-Based Service Migration, ScienceDirect
6. Lundberg & Lee (2017) — SHAP: A Unified Approach to Interpreting Model Predictions

---

*Auburn University at Montgomery | Computer Science | April 2026*
