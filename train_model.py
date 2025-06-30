import pandas as pd
import joblib
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix, roc_curve
import matplotlib.pyplot as plt

from utils.preprocess import load_data, initial_inspection, identify_columns, build_preprocessing_pipeline

def main():
    # Load dataset
    df = load_data('data/loan_data_2007_2014.csv')
    # Initial inspection
    initial_inspection(df)

    # TODO: Define target variable based on domain knowledge and data dictionary
    # Example: assume 'loan_status' column exists and values can be mapped to GOOD/BAD
    # df['target'] = df['loan_status'].apply(lambda x: 1 if x in ['Charged Off', 'Default'] else 0)

    # For demonstration, replace with actual logic:
    # target_col = '...'
    # df['target'] = ...

    # Drop rows with missing target
    # df = df.dropna(subset=['target'])

    # Identify features
    numeric_cols, categorical_cols = identify_columns(df.drop(columns=['target'], errors='ignore'))

    # Build preprocessing pipeline
    preprocessor = build_preprocessing_pipeline(numeric_cols, categorical_cols)

    # Split data
    X = df.drop(columns=['target'])
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    # Preprocess
    X_train_proc = preprocessor.fit_transform(X_train)
    X_test_proc = preprocessor.transform(X_test)

    # Models to train
    models = {
        'LogisticRegression': LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42),
        'RandomForest': RandomForestClassifier(class_weight='balanced', random_state=42)
    }

    # Hyperparameter grids
    param_grids = {
        'LogisticRegression': {
            'C': [0.01, 0.1, 1, 10]
        },
        'RandomForest': {
            'n_estimators': [100, 200],
            'max_depth': [None, 10, 20]
        }
    }

    best_estimators = {}
    for name, model in models.items():
        print(f"Training {name}...")
        grid = GridSearchCV(model, param_grids[name], cv=5, scoring='roc_auc', n_jobs=-1)
        grid.fit(X_train_proc, y_train)
        best_estimators[name] = grid.best_estimator_
        print(f"Best params for {name}: {grid.best_params_}")
        # Evaluate
        y_pred = grid.predict(X_test_proc)
        y_proba = grid.predict_proba(X_test_proc)[:, 1]
        print(f"Classification Report for {name}:")
        print(classification_report(y_test, y_pred))
        print("ROC-AUC:", roc_auc_score(y_test, y_proba))
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print("Confusion Matrix:")
        print(cm)
        # ROC Curve
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        plt.figure()
        plt.plot(fpr, tpr, label=f'{name} (AUC = {roc_auc_score(y_test, y_proba):.2f})')
        plt.plot([0,1], [0,1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {name}')
        plt.legend()
        plt.show()

    # Save best model (example: choose LogisticRegression)
    best_model = best_estimators['LogisticRegression']
    joblib.dump(best_model, 'models/credit_risk_model.pkl')
    print("Model saved to models/credit_risk_model.pkl")

if __name__ == '__main__':
    main()