import pandas as pd

def load_dataset(filepath):
    """
    Load the phishing dataset from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
    
    Returns:
        pd.DataFrame: Loaded dataset
    """
    df = pd.read_csv(filepath)
    print(f"Dataset shape: {df.shape}")
    print(f"\nFirst few rows:\n{df.head()}")
    print(f"\nDataset info:\n{df.info()}")
    return df
