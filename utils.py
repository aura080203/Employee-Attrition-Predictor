#!/usr/bin/env python3
"""
utils.py — Shared feature engineering for training and inference.
Import this in both employee_attrition_project.py and app.py.
"""
import pandas as pd
import numpy as np


# Columns to drop on load — constants shared across train/serve
DROP_COLS = ['EmployeeCount', 'EmployeeNumber', 'Over18', 'StandardHours']


def load_data(url: str) -> pd.DataFrame:
    """Load raw IBM HR dataset and drop constant/ID columns."""
    df = pd.read_csv(url)
    df.drop(columns=DROP_COLS, inplace=True, errors='ignore')
    return df


def encode_target(df: pd.DataFrame) -> pd.DataFrame:
    """Convert Attrition column from Yes/No string to int (1/0)."""
    df = df.copy()
    if df['Attrition'].dtype == object:
        df['Attrition'] = (df['Attrition'] == 'Yes').astype(int)
    return df


def add_base_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add engineered features that do NOT require label information.
    Safe to use on both training and inference data without leakage.
    """
    df = df.copy()
    ot_binary = (df['OverTime'] == 'Yes').astype(int)
    df['OT_x_JobSat']     = ot_binary * df['JobSatisfaction']
    df['TenureRatio']     = df['YearsSinceLastPromotion'] / (df['YearsAtCompany'] + 1)
    df['AvgSatisfaction'] = df[['EnvironmentSatisfaction',
                                 'JobSatisfaction',
                                 'RelationshipSatisfaction']].mean(axis=1)
    df['YearsPerCompany'] = df['YearsAtCompany'] / (df['NumCompaniesWorked'] + 1)
    return df


def add_role_attrition_rate(df: pd.DataFrame,
                             role_rates: dict,
                             global_mean: float) -> pd.DataFrame:
    """
    Add RoleAttritionRate using a pre-computed mapping from TRAINING data only.
    Unseen roles fall back to global_mean to avoid leakage.

    Args:
        df:          DataFrame to add the feature to (train or test)
        role_rates:  dict mapping JobRole -> mean attrition rate (from train only)
        global_mean: fallback rate for roles not seen during training
    """
    df = df.copy()
    df['RoleAttritionRate'] = df['JobRole'].map(role_rates).fillna(global_mean)
    return df


def engineer_features_train(df: pd.DataFrame):
    """
    Full feature engineering for TRAINING data.
    Returns (df_engineered, role_rates, global_mean) so the rate map
    can be saved and reused at inference time without leakage.
    """
    df = encode_target(df)
    df = add_base_features(df)

    # Compute role rates from training labels ONLY
    role_rates  = df.groupby('JobRole')['Attrition'].mean().to_dict()
    global_mean = df['Attrition'].mean()
    df = add_role_attrition_rate(df, role_rates, global_mean)

    return df, role_rates, global_mean


def engineer_features_inference(df: pd.DataFrame,
                                 role_rates: dict,
                                 global_mean: float) -> pd.DataFrame:
    """
    Feature engineering for INFERENCE (test set or live app).
    Uses role_rates computed from training data — no leakage.
    """
    df = add_base_features(df)
    df = add_role_attrition_rate(df, role_rates, global_mean)
    return df
