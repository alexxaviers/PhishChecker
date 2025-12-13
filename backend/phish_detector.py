"""
Phishing URL Detection System

Main script that:
1. Loads the phishing dataset
2. Preprocesses features (drops non-numeric columns, scales features)
3. Splits data into train/test (80/20)
4. Trains three models with GridSearchCV hyperparameter tuning
5. Evaluates all models
6. Saves results to markdown table
7. Identifies and prints best performing model
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data_loader import load_dataset
from src.preprocess import preprocess_data
from src.model_training import train_logistic_regression, train_random_forest, train_xgboost
from src.evaluate import evaluate_model, generate_results_table


def main():
    """
    Main execution function for the phishing detector.
    """
    print("="*80)
    print("PHISHING URL DETECTION SYSTEM")
    print("="*80)
    
    # Step 1: Load dataset
    print("\n[Step 1] Loading dataset...")
    dataset_path = os.path.join(os.path.dirname(__file__), 'data', 'PhiUSIIL_Phishing_URL_Dataset.csv')
    df = load_dataset(dataset_path)
    
    # Step 2: Preprocess data
    print("\n[Step 2] Preprocessing data...")
    X, y, scaler = preprocess_data(df)
    
    # Step 3: Split data into train/test (80/20)
    print("\n[Step 3] Splitting data into train/test sets (80/20)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Training set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")
    
    # Step 4: Train models with GridSearchCV
    print("\n[Step 4] Training models with hyperparameter tuning...")
    
    print("\n--- Logistic Regression ---")
    lr_model, lr_params, lr_cv_score = train_logistic_regression(X_train, y_train)
    
    print("\n--- Random Forest ---")
    rf_model, rf_params, rf_cv_score = train_random_forest(X_train, y_train)
    
    print("\n--- XGBoost ---")
    xgb_model, xgb_params, xgb_cv_score = train_xgboost(X_train, y_train)
    
    # Step 5: Evaluate all models
    print("\n[Step 5] Evaluating models on test set...")
    
    lr_results = evaluate_model(lr_model, X_test, y_test, "Logistic Regression")
    rf_results = evaluate_model(rf_model, X_test, y_test, "Random Forest")
    xgb_results = evaluate_model(xgb_model, X_test, y_test, "XGBoost")
    
    all_results = [lr_results, rf_results, xgb_results]
    
    # Step 6: Save results to markdown table
    print("\n[Step 6] Saving results to markdown table...")
    report_path = os.path.join(os.path.dirname(__file__), 'report', 'results.md')
    generate_results_table(all_results, report_path)
    
    # Step 7: Identify and print best model
    print("\n[Step 7] Determining best model...")
    best_model = max(all_results, key=lambda x: x['F1-Score'])
    print(f"\n{'='*80}")
    print(f"BEST PERFORMING MODEL: {best_model['Model']}")
    print(f"F1-Score: {best_model['F1-Score']:.4f}")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
