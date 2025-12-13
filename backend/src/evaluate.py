from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def evaluate_model(model, X_test, y_test, model_name):
    """
    Evaluate a model and return metrics.
    
    Args:
        model: Trained model
        X_test: Test features
        y_test: Test labels
        model_name (str): Name of the model
    
    Returns:
        dict: Dictionary containing accuracy, precision, recall, f1_score
    """
    y_pred = model.predict(X_test)
    
    results = {
        'Model': model_name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred, average='weighted'),
        'Recall': recall_score(y_test, y_pred, average='weighted'),
        'F1-Score': f1_score(y_test, y_pred, average='weighted')
    }
    
    print(f"\n{model_name} Results:")
    print(f"  Accuracy:  {results['Accuracy']:.4f}")
    print(f"  Precision: {results['Precision']:.4f}")
    print(f"  Recall:    {results['Recall']:.4f}")
    print(f"  F1-Score:  {results['F1-Score']:.4f}")
    
    return results


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
