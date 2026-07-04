#!/usr/bin/env python3
"""
employee_attrition_project.py
─────────────────────────────
Trains a calibrated stacking ensemble for HR attrition prediction.

"""
import logging
import json
from pathlib import Path

import pandas as pd
import numpy as np
from sklearn.model_selection import (
    train_test_split, StratifiedKFold, GridSearchCV
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, classification_report,
    roc_auc_score, precision_recall_curve, auc, confusion_matrix
)
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
import joblib

from utils import load_data, engineer_features_train, engineer_features_inference

# ── Data URL ────────────────────────────────────────────────────────────────
DATA_URL = (
    'https://raw.githubusercontent.com/'
    'mragpavank/ibm-hr-analytics-attrition-dataset/'
    'master/WA_Fn-UseC_-HR-Employee-Attrition.csv'
)


def setup_logging():
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )


def split_data(df: pd.DataFrame, test_size=0.2, random_state=42):
    X = df.drop('Attrition', axis=1)
    y = df['Attrition']
    return train_test_split(
        X, y,
        test_size=test_size,
        stratify=y,
        random_state=random_state
    )


def build_pipeline(numeric_features, categorical_features) -> ImbPipeline:
    preprocessor = ColumnTransformer([
        ('num', StandardScaler(), numeric_features),
        ('cat', OneHotEncoder(handle_unknown='ignore', sparse_output=False),
         categorical_features)
    ])

    xgb     = XGBClassifier(eval_metric='auc', random_state=42,
                             scale_pos_weight=5, verbosity=0)
    lgb     = LGBMClassifier(random_state=42, is_unbalance=True, verbose=-1)
    rf      = RandomForestClassifier(random_state=42, class_weight='balanced')
    lr_base = LogisticRegression(max_iter=1000, class_weight='balanced',
                                  random_state=42)

    stack = StackingClassifier(
        estimators=[('xgb', xgb), ('lgb', lgb), ('rf', rf), ('lr', lr_base)],
        final_estimator=LogisticRegression(class_weight='balanced'),
        cv=3,       # reduced from 5 to speed up GridSearchCV
        n_jobs=-1
    )

    return ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.5)),
        ('tomek', TomekLinks()),
        ('stack', stack)
    ])


def optimize_hyperparameters(pipeline, X_train, y_train) -> GridSearchCV:
    """
    Reduced grid vs original: 8 combinations × 3 folds = 24 fits.
    Original was 32 combinations × 5 folds = 160 fits.
    Still covers the most impactful params (XGB depth, RF trees).
    """
    param_grid = {
        'stack__xgb__n_estimators': [100, 200],
        'stack__xgb__max_depth':    [3, 5],
        'stack__rf__n_estimators':  [100, 200],
    }
    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
    grid = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        scoring='roc_auc',
        cv=cv,
        n_jobs=-1,
        verbose=2,          # shows progress so you know it's running
        error_score='raise'
    )
    logging.info("Starting GridSearchCV — this will take a few minutes...")
    grid.fit(X_train, y_train)
    logging.info(f"Best params: {grid.best_params_}")
    logging.info(f"Best CV ROC AUC: {grid.best_score_:.4f}")
    return grid


def calibrate_model(model, X_train, y_train) -> CalibratedClassifierCV:
    """Isotonic calibration on training data for reliable probability estimates."""
    calib = CalibratedClassifierCV(model, method='isotonic', cv=3)
    calib.fit(X_train, y_train)
    return calib


def compute_cost_threshold(y_true, y_proba, cost_fp=1, cost_fn=5):
    """
    Find the threshold that minimises total cost where:
      - False Negative (missing at-risk employee) costs 5x more than
      - False Positive (flagging someone who stays)
    """
    thresholds = np.linspace(0, 1, 101)
    best_thr, best_cost = 0, np.inf
    for t in thresholds:
        y_pred = (y_proba >= t).astype(int)
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        cost = fp * cost_fp + fn * cost_fn
        if cost < best_cost:
            best_cost, best_thr = cost, t
    return best_thr, best_cost


def evaluate_model(model, X_test, y_test) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    y_pred  = model.predict(X_test)

    acc     = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    precision, recall, pr_thresholds = precision_recall_curve(y_test, y_proba)
    pr_auc  = auc(recall, precision)

    logging.info(f"Accuracy @0.5:  {acc:.4f}")
    logging.info(f"ROC AUC:        {roc_auc:.4f}")
    logging.info(f"PR AUC:         {pr_auc:.4f}")
    logging.info("\n" + classification_report(y_test, y_pred))

    # F1-optimal threshold
    f1      = 2 * precision * recall / (precision + recall + 1e-12)
    idx_f1  = np.nanargmax(f1[:-1])
    thr_f1  = pr_thresholds[idx_f1]
    logging.info(f"F1-optimal threshold:     {thr_f1:.3f}")

    # Recall >= 51% threshold
    idx_r   = np.where(recall >= 0.51)[0]
    thr_r51 = pr_thresholds[idx_r[0]] if len(idx_r) else thr_f1
    logging.info(f"Recall>=51% threshold:    {thr_r51:.3f}")

    # Cost-sensitive threshold
    thr_cost, cost_val = compute_cost_threshold(y_test, y_proba)
    logging.info(f"Cost-optimal threshold:   {thr_cost:.2f} (cost={cost_val})")

    # Max accuracy threshold
    grid_thr = np.linspace(0, 1, 101)
    accs     = [(t, accuracy_score(y_test, (y_proba >= t).astype(int)))
                for t in grid_thr]
    thr_acc, best_acc = max(accs, key=lambda x: x[1])
    logging.info(f"Max-accuracy threshold:   {thr_acc:.2f} ({best_acc:.4f})")

    return {
        'threshold_f1':       float(thr_f1),
        'threshold_51recall': float(thr_r51),
        'threshold_maxacc':   float(thr_acc),
        'threshold_cost':     float(thr_cost),
        # Store final metrics for reference
        'metrics': {
            'accuracy':  round(acc, 4),
            'roc_auc':   round(roc_auc, 4),
            'pr_auc':    round(pr_auc, 4),
        }
    }


def save_artifacts(model, thresholds, role_rates, global_mean,
                   out_dir='artifacts'):
    """
    Save model + thresholds + role_rates map.
    role_rates and global_mean must be saved so the app can
    compute RoleAttritionRate at inference time without leakage.
    """
    path = Path(out_dir)
    path.mkdir(exist_ok=True)

    joblib.dump(model, path / 'attrition_model.pkl')

    with open(path / 'thresholds.json', 'w') as f:
        json.dump(thresholds, f, indent=2)

    # Save role rates map so app.py can use it at inference time
    with open(path / 'role_rates.json', 'w') as f:
        json.dump({'role_rates': role_rates, 'global_mean': global_mean}, f,
                  indent=2)

    logging.info(f"All artifacts saved to {path.resolve()}/")
    logging.info(f"  attrition_model.pkl")
    logging.info(f"  thresholds.json")
    logging.info(f"  role_rates.json")


if __name__ == '__main__':
    setup_logging()

    # 1. Load raw data
    logging.info("Loading data...")
    df_raw = load_data(DATA_URL)

    # 2. Engineer features on full dataset first to get role_rates from train
    #    split only — fixes the data leakage bug in the original code
    logging.info("Splitting data...")
    df_encoded = df_raw.copy()
    df_encoded['Attrition'] = (df_encoded['Attrition'] == 'Yes').astype(int)
    X_raw, X_te_raw, y_tr, y_te = train_test_split(
        df_encoded.drop('Attrition', axis=1),
        df_encoded['Attrition'],
        test_size=0.2,
        stratify=df_encoded['Attrition'],
        random_state=42
    )

    # Reconstruct train df to compute role rates from train labels only
    df_train_raw = X_raw.copy()
    df_train_raw['Attrition'] = y_tr.values

    logging.info("Engineering features (train only — no leakage)...")
    df_train, role_rates, global_mean = engineer_features_train(df_train_raw)

    # Apply same transforms to test using TRAIN-computed role rates
    df_test = engineer_features_inference(X_te_raw, role_rates, global_mean)

    X_tr = df_train.drop('Attrition', axis=1)
    y_tr = df_train['Attrition']
    X_te = df_test

    # 3. Build and tune pipeline
    num_feats = X_tr.select_dtypes(include='number').columns.tolist()
    cat_feats = X_tr.select_dtypes(include='object').columns.tolist()
    logging.info(f"Features: {len(num_feats)} numeric, {len(cat_feats)} categorical")

    pipeline = build_pipeline(num_feats, cat_feats)
    gridcv   = optimize_hyperparameters(pipeline, X_tr, y_tr)
    best     = gridcv.best_estimator_

    # 4. Calibrate on training data
    logging.info("Calibrating model...")
    calib = calibrate_model(best, X_tr, y_tr)

    # 5. Evaluate on held-out test set
    logging.info("Evaluating on test set...")
    thresholds = evaluate_model(calib, X_te, y_te)

    # 6. Save everything
    save_artifacts(calib, thresholds, role_rates, global_mean)
    logging.info("Done. Run: streamlit run app.py")
