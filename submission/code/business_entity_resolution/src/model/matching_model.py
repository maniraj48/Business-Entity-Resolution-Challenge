"""
Matching ML Model module for Business Entity Resolution Challenge.
Provides train, save, load, predict_proba, and threshold prediction interface.
Supports Logistic Regression and HistGradientBoosting / Random Forest classifiers.
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any, Optional
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

class MatchingModel:
    def __init__(self, model_type: str = "hist_gb", random_state: int = 42, **kwargs):
        """
        Initialize Matching Classifier Model.
        
        :param model_type: Classifier choice ('logistic', 'hist_gb', 'random_forest')
        :param random_state: Random seed for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.kwargs = kwargs
        self.feature_names: List[str] = []
        self.classifier = self._build_classifier()

    def _build_classifier(self):
        if self.model_type == "logistic":
            base_clf = LogisticRegression(
                C=self.kwargs.get("C", 1.0),
                max_iter=1000,
                class_weight="balanced",
                random_state=self.random_state
            )
            return Pipeline([
                ("scaler", StandardScaler()),
                ("clf", base_clf)
            ])
        elif self.model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=self.kwargs.get("n_estimators", 100),
                max_depth=self.kwargs.get("max_depth", 10),
                class_weight="balanced",
                random_state=self.random_state,
                n_jobs=-1
            )
        elif self.model_type == "hist_gb":
            return HistGradientBoostingClassifier(
                max_iter=self.kwargs.get("max_iter", 150),
                learning_rate=self.kwargs.get("learning_rate", 0.05),
                max_depth=self.kwargs.get("max_depth", 6),
                class_weight="balanced",
                random_state=self.random_state
            )
        else:
            raise ValueError(f"Unknown model_type: {self.model_type}")

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "MatchingModel":
        """Fit model on training feature matrix X and labels y."""
        self.feature_names = list(X.columns)
        X_mat = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
        self.classifier.fit(X_mat, y.values)
        return self

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Predict positive class match probability."""
        X_mat = np.nan_to_num(X.values, nan=0.0, posinf=0.0, neginf=0.0)
        clf = self.classifier
        # For Pipeline, get the final estimator's classes_
        classes = clf.classes_ if hasattr(clf, 'classes_') else clf[-1].classes_
        prob = clf.predict_proba(X_mat)
        if 1 in classes:
            idx = list(classes).index(1)
            return prob[:, idx]
        # Model never saw class 1 — return zeros
        return np.zeros(prob.shape[0])

    def predict_binary(self, X: pd.DataFrame, threshold: float = 0.5) -> np.ndarray:
        """Predict binary match decision using probability threshold."""
        probs = self.predict_proba(X)
        return (probs >= threshold).astype(int)

    def save(self, filepath: str) -> None:
        """Save model instance to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        artifact = {
            "model_type": self.model_type,
            "feature_names": self.feature_names,
            "classifier": self.classifier
        }
        joblib.dump(artifact, filepath)
        print(f" -> Model artifact saved to: {filepath}")

    @classmethod
    def load(cls, filepath: str) -> "MatchingModel":
        """Load model instance from disk."""
        artifact = joblib.load(filepath)
        model = cls(model_type=artifact["model_type"])
        model.feature_names = artifact["feature_names"]
        model.classifier = artifact["classifier"]
        return model

def evaluate_pair_classifier(
    model: MatchingModel,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Evaluate pair-level classification metrics."""
    probs = model.predict_proba(X_val)
    preds = (probs >= threshold).astype(int)
    
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_val, preds, average="binary", zero_division=0
    )
    f05 = (1.25 * prec * rec) / (0.25 * prec + rec) if (0.25 * prec + rec) > 0 else 0.0
    
    unique_classes = np.unique(y_val)
    if len(unique_classes) > 1:
        auc = float(roc_auc_score(y_val, probs))
    else:
        auc = float("nan")  # ROC-AUC is undefined when y_val has only 1 class
    
    return {
        "pair_precision": float(prec),
        "pair_recall": float(rec),
        "pair_f1": float(f1),
        "pair_f05": float(f05),
        "pair_roc_auc": auc
    }
