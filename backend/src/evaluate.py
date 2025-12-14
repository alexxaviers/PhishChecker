from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_curve, auc, roc_auc_score
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model and return metrics, confusion matrix, and classification report.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1_score,
              confusion_matrix, and classification_report
    """
    y_pred = model.predict(X_test)
    
    # Compute metrics
    cm = confusion_matrix(y_test, y_pred)
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted'),
        'Confusion_Matrix': cm,
        'Classification_Report': class_report
    }
    
    
    print(f"\n  Confusion Matrix:\n{cm}")
    print(f"\n  Classification Report:\n{classification_report(y_test, y_pred)}")
    
    return results


def plot_confusion_matrix(model_name, cm, save_path):
    """
    Plot and save a confusion matrix as a heatmap.
    
    Args:
        model_name (str): Name of the model
        cm (np.ndarray): Confusion matrix from sklearn.metrics.confusion_matrix
        save_path (str): Path to save the figure (e.g., '/path/to/confusion_matrix_model.png')
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=True,
                xticklabels=['Legitimate', 'Phishing'],
                yticklabels=['Legitimate', 'Phishing'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Confusion matrix plot saved to: {save_path}")


def generate_results_table(results_list, output_path):
    """
    Generate a markdown table with model results.
    
    Args:
        results_list (list): List of result dictionaries
        output_path (str): Path to save the markdown file
    """
    # Create markdown table
    markdown_content = "# Phishing URL Detection - Model Evaluation Results\n\n"
    markdown_content += "| Model | Accuracy | Precision | Recall | F1-Score |\n"
    markdown_content += "|-------|----------|-----------|--------|----------|\n"
    
    for result in results_list:
        markdown_content += (
            f"| {result['Model']} | "
            f"{result['Accuracy']:.4f} | "
            f"{result['Precision']:.4f} | "
            f"{result['Recall']:.4f} | "
            f"{result['F1-Score']:.4f} |\n"
        )
    
    # Write to file
    with open(output_path, 'w') as f:
        f.write(markdown_content)
    
    print(f"\nResults table saved to: {output_path}")


def plot_feature_importances(model, feature_names, model_name, save_path):
    """
    Plot and save feature importances as a horizontal bar chart.
    
    Supports:
    - RandomForestClassifier: model.feature_importances_
    - XGBClassifier: model.feature_importances_
    - LogisticRegression: abs(model.coef_[0])
    
    Args:
        model: Trained model (RandomForest, XGBoost, or LogisticRegression)
        feature_names (list or array): List of feature names
        model_name (str): Name of the model
        save_path (str): Path to save the figure (e.g., '/path/to/feature_importances_model.png')
    """
    # Extract feature importances
    if hasattr(model, 'feature_importances_'):
        # RandomForestClassifier, XGBClassifier
        importances = model.feature_importances_
    elif hasattr(model, 'coef_'):
        # LogisticRegression
        importances = np.abs(model.coef_[0])
    else:
        print(f"  Model {model_name} does not have extractable feature importances.")
        return
    
    # Create DataFrame with feature names and importances
    feature_importance_df = pd.DataFrame({
        'Feature': feature_names,
        'Importance': importances
    })
    
    # Sort by importance (descending) and select top 15
    feature_importance_df = feature_importance_df.sort_values(
        'Importance', ascending=True
    ).tail(15)
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='steelblue')
    plt.xlabel('Importance Score')
    plt.title(f'Top 15 Feature Importances - {model_name}')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  Feature importance plot saved to: {save_path}")


def plot_roc_curve(model, X_test, y_test, model_name, save_path):
    """
    Plot and save ROC curve with AUC score.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
        save_path (str): Path to save the figure (e.g., '/path/to/roc_model.png')
    """
    # Get predicted probabilities for positive class
    try:
        y_pred_proba = model.predict_proba(X_test)[:, 1]
    except AttributeError:
        print(f"  Model {model_name} does not support predict_proba(). Skipping ROC curve.")
        return
    
    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2.5, label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve - {model_name}')
    plt.legend(loc='lower right')
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  ROC curve plot saved to: {save_path} (AUC = {roc_auc:.4f})")
