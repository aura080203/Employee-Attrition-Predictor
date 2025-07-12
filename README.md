# Employee-Attrition-Predictor
Custom features enriched a public HR attrition dataset, then an SMOTE/Tomek-preprocessed stacking ensemble (XGBoost, LightGBM, RandomForest, LogisticRegression) was trained. GridSearchCV tuning and isotonic calibration, plus threshold optimization for F1, cost, and accuracy, yielded 86.7% accuracy, 0.807 ROC AUC, and 0.520 PR AUC.
