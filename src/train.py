from pathlib import Path
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import ( ConfusionMatrixDisplay, RocCurveDisplay,
                             classification_report,
                             f1_score,roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

processed_dir = Path('data/processed')
models_dir = Path('models')
reports_dir = Path('reports')

#features that will be passed ot Standard scaler library
numeric_features = [
    'GridPosition','avg_air_temp','avg_track_temp',
    'driver_dnf_rate_last5',
    'team_dnf_rate_last5',
    'circuit_dnf_rate_hist'
]


#features that will be passed to one shot encoder to encode them
categorical_features = ['TeamName', "rain"]

train_seasons = [2022,2023,2024]
test_season = 2025

def load_features():
    path = processed_dir / "features.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run build_features.py first.")
    
    return pd.read_csv(path)

def build_pipeline():
    # Numeric pipeline: Impute missing values with median, then scale
    num_transformer = make_pipeline(
        SimpleImputer(strategy="median"),
        StandardScaler()
    )
    
    # Categorical pipeline: One-hot encode strings
    cat_transformer = OneHotEncoder(handle_unknown="ignore")

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", num_transformer, numeric_features),
            ("cat", cat_transformer, categorical_features),
        ]
    )

    model = LogisticRegression(class_weight="balanced", max_iter=1000)
    return Pipeline(steps=[("preprocess", preprocessor), ("model", model)])


def evaluate(pipeline, X_test, y_test):
    reports_dir.mkdir(parents=True, exist_ok=True)
    
    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]
    
    print("\n=== Classification report (test season {}) ===".format(test_season))
    print(classification_report(y_test, y_pred, target_names=["Finished", "DNF"]))
    print("F1 score (DNF class):", round(f1_score(y_test, y_pred), 3))
    print("ROC-AUC:", round(roc_auc_score(y_test, y_proba), 3))
    
                ########confusion matix
            # ################ACTUAL RESULTS
                #             Finished   |   DNF
                #         +------------+------------+
            # Predicted   |   True     |   False    |
            #Finished     | Negative   | Negative   |
            #             +------------+------------+
            # Predicted   |   False    |   True     |
            #DNF          | Positive   | Positive   |
            #             +------------+------------+
    
    fig, ax = plt.subplots(figsize=(5, 5))
    ConfusionMatrixDisplay.from_predictions(
        y_test,y_pred, display_labels=["Finished", "DNF"], ax=ax
    )
    fig.savefig(reports_dir / "confusion_matrix.png", bbox_inches="tight")
    plt.close(fig)
    
    #Reciever Operator Characteristic(ROC)
    #to find the best threshold
    #instead of getting confused by multiple confusion matrix
    #PUN INTENDED
    fig, ax = plt.subplots(figsize=(5, 5))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    fig.savefig(reports_dir / "roc_curve.png", bbox_inches="tight")
    plt.close(fig)
    
    
    #calibration curve to determine how reliable our probalities is
    prob_true, prob_pred = calibration_curve(y_test, y_proba, n_bins=10, strategy='quantile')
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.plot(prob_pred, prob_true, marker="o", label="model")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray", label="Perfectly calibrated")
    ax.set_xlabel("Predicted DNF probability")
    ax.set_ylabel("Observed DNF frequency")
    ax.set_title("Calibration curve")
    ax.legend()
    fig.savefig(reports_dir / "calibration_curve.png", bbox_inches="tight")
    plt.close(fig)

    print(f"\nSaved plots to {reports_dir}/")
#determine which features affects DNF positively or negatively    
def print_coefficients(pipeline):
    model = pipeline.named_steps['model']
    preprocessor = pipeline.named_steps['preprocess']
    feature_names = preprocessor.get_feature_names_out()
    coefs = pd.Series(model.coef_[0], index=feature_names).sort_values()
    print("\n=== Logistic regression coefficients (higher = more DNF risk) ===")
    print(coefs)
    
def main():
    
    df = load_features()
    train_df = df[df['season'].isin(train_seasons)]  
    test_df = df[df['season'] == test_season]        
    
    if train_df.empty or test_df.empty:
        raise ValueError(
            f"Train or test split is empty. Check that features.csv contains "
            f"seasons {train_seasons} and {test_season}."
        )
   
    all_features = numeric_features + categorical_features
    X_train, y_train = train_df[all_features], train_df['is_dnf']
    X_test, y_test = test_df[all_features], test_df['is_dnf']
    
    print(f"Train: {len(X_train)} rows ({train_seasons}) | Test: {len(X_test)} rows ({test_season})")
    print(f"Train DNF rate: {y_train.mean():.3f} | Test DNF rate: {y_test.mean():.3f}")
    
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)
    
    evaluate(pipeline, X_test, y_test)
    print_coefficients(pipeline)

    models_dir.mkdir(parents=True, exist_ok=True)
    model_path = models_dir / "dnf_model.pkl"
    joblib.dump(pipeline, model_path)
    print(f"\nSaved trained pipeline to {model_path}")
    
    
    
if __name__ == "__main__":
        main()