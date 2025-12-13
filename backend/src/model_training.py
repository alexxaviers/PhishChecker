from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV

def train_logistic_regression(X_train, y_train):
    """
    Train Logistic Regression with GridSearchCV hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        tuple: (best_model, best_params, cv_score)
    """
    param_grid = {
        'C': [0.1, 1, 10]
    }
    
    lr = LogisticRegression(max_iter=1000, random_state=42)
    grid_search = GridSearchCV(lr, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Logistic Regression - Best params: {grid_search.best_params_}")
    print(f"Logistic Regression - Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_random_forest(X_train, y_train):
    """
    Train Random Forest with GridSearchCV hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        tuple: (best_model, best_params, cv_score)
    """
    param_grid = {
        'n_estimators': [100, 300],
        'max_depth': [None, 10, 20]
    }
    
    rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"Random Forest - Best params: {grid_search.best_params_}")
    print(f"Random Forest - Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_


def train_xgboost(X_train, y_train):
    """
    Train XGBoost with GridSearchCV hyperparameter tuning.
    
    Args:
        X_train: Training features
        y_train: Training labels
    
    Returns:
        tuple: (best_model, best_params, cv_score)
    """
    param_grid = {
        'n_estimators': [200, 400],
        'max_depth': [4, 6, 8]
    }
    
    xgb = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    grid_search = GridSearchCV(xgb, param_grid, cv=5, scoring='f1_weighted', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    
    print(f"XGBoost - Best params: {grid_search.best_params_}")
    print(f"XGBoost - Best CV score: {grid_search.best_score_:.4f}")
    
    return grid_search.best_estimator_, grid_search.best_params_, grid_search.best_score_
