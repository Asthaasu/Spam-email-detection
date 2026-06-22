import os
import sys
import pickle
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

# Allow this file to import gmail_features.py from the same src folder
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from gmail_features import prepare_training_data


# Project root folder
BASE_DIR = os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))
)

MODEL_DIR = os.path.join(BASE_DIR, "model")
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "baseline_rf.pkl"
)


def train_baseline(X, y):
    """Train and evaluate the Phase 3 Random Forest baseline."""

    # 1. Stratified train-test split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training samples: {len(X_train)}")
    print(f"Testing samples: {len(X_test)}")

    # 2. Train baseline Random Forest
    model = RandomForestClassifier(
        n_estimators=100,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )

    model.fit(X_train, y_train)

    # 3. Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # 4. Metrics
    metrics = {
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, y_pred_proba),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # 5. Five-fold cross-validation on training data
    cv_scores = cross_val_score(
        model,
        X_train,
        y_train,
        cv=5,
        scoring="f1",
    )

    metrics["cv_f1_mean"] = cv_scores.mean()
    metrics["cv_f1_std"] = cv_scores.std()

    print("\n" + "=" * 45)
    print("BASELINE MODEL (Random Forest)")
    print("=" * 45)

    print(f"Precision: {metrics['precision']:.3f}")
    print(f"Recall:    {metrics['recall']:.3f}")
    print(f"F1:        {metrics['f1']:.3f}")
    print(f"ROC-AUC:   {metrics['roc_auc']:.3f}")
    print(
        f"CV F1:     {metrics['cv_f1_mean']:.3f} "
        f"± {metrics['cv_f1_std']:.3f}"
    )

    print("\nConfusion Matrix:")
    print(np.array(metrics["confusion_matrix"]))

    return model, metrics


def main():
    # Load raw Gmail data and create Phase 2 features
    X, y, feature_names = prepare_training_data()

    print("\nPHASE 3: TRAINING BASELINE MODEL")
    print(f"Feature matrix shape: {X.shape}")
    print(f"Spam ratio: {y.mean():.2%}\n")

    model, metrics = train_baseline(X, y)

    # Save model and exact feature order
    os.makedirs(MODEL_DIR, exist_ok=True)

    with open(MODEL_PATH, "wb") as file:
        pickle.dump(model, file)

    feature_path = os.path.join(
        MODEL_DIR,
        "baseline_feature_names.pkl",
    )

    with open(feature_path, "wb") as file:
        pickle.dump(list(feature_names), file)

    print("\nModel saved to:")
    print(MODEL_PATH)

    print("\nFeature names saved to:")
    print(feature_path)


if __name__ == "__main__":
    main()
