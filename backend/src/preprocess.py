import pandas as pd
from sklearn.preprocessing import StandardScaler

def preprocess_data(df):
    """
    Preprocess the phishing dataset.
    
    Steps:
    1. Drop non-numeric columns
    2. Fill NaN values
    3. Separate features and label
    4. Scale numerical features
    
    Args:
        df (pd.DataFrame): Raw dataset
    
    Returns:
        tuple: (X_scaled, y, scaler) where:
            - X_scaled: Scaled feature matrix
            - y: Target labels
            - scaler: Fitted StandardScaler object
    """
    # Create a copy to avoid modifying original
    df = df.copy()
    
    # Drop non-numeric columns as specified
    columns_to_drop = ["FILENAME", "URL", "Domain", "TLD", "Title"]
    df = df.drop(columns=columns_to_drop, errors='ignore')
    
    # Fill NaN values with median
    df = df.fillna(df.median(numeric_only=True))
    
    # Separate features (X) and target (y)
    y = df['label']
    X = df.drop('label', axis=1)
    
    # Scale features using StandardScaler
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Convert scaled array back to DataFrame to preserve column names
    X_scaled = pd.DataFrame(X_scaled, columns=X.columns)
    
    print(f"Features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
    print(f"Number of features: {X_scaled.shape[1]}")
    
    return X_scaled, y, scaler
