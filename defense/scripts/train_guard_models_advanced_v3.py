#!/usr/bin/env python
"""
Advanced training for three AEA guard classifiers:

1. Regularized Logistic Regression with internal hyperparameter search
2. Random Forest with many trees and internal hyperparameter search
3. Deep MLP ensemble with 5-fold out-of-fold training, dropout, batch
   normalization, AdamW, learning-rate scheduling, gradient clipping,
   class weighting, and early stopping

Important:
- Hyperparameters and thresholds are chosen using only the training set.
- The fixed test set is evaluated only after model selection.
- The script is designed for the harder semantic feature file produced by
  create_guard_features_hard_v3.py.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import random
import warnings
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.exceptions import ConvergenceWarning
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_npz(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(f"Missing feature file: {path.resolve()}")
    data = np.load(path, allow_pickle=False)
    return (
        data["X"].astype(np.float32),
        data["y"].astype(np.int64),
        data["ids"].astype(str),
    )


def select_threshold(
    y_true: np.ndarray,
    score: np.ndarray,
    minimum_recall: float | None = None,
) -> tuple[float, dict[str, float]]:
    best_threshold = 0.5
    best_tuple = (-1.0, -1.0, -1.0)

    for threshold in np.linspace(0.05, 0.95, 181):
        pred = (score >= threshold).astype(int)
        precision = precision_score(y_true, pred, zero_division=0)
        recall = recall_score(y_true, pred, zero_division=0)
        f1 = f1_score(y_true, pred, zero_division=0)

        if minimum_recall is not None and recall < minimum_recall:
            continue

        candidate = (f1, recall, precision)
        if candidate > best_tuple:
            best_tuple = candidate
            best_threshold = float(threshold)

    if best_tuple[0] < 0:
        return select_threshold(y_true, score, minimum_recall=None)

    return best_threshold, {
        "validation_f1": float(best_tuple[0]),
        "validation_recall": float(best_tuple[1]),
        "validation_precision": float(best_tuple[2]),
    }


def evaluate(
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    pred = (score >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()

    result = {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
        "false_positive_rate": float(fp / (fp + tn)) if fp + tn else 0.0,
        "false_negative_rate": float(fn / (fn + tp)) if fn + tp else 0.0,
        "roc_auc": None,
        "pr_auc": None,
        "classification_report": classification_report(
            y_true,
            pred,
            labels=[0, 1],
            target_names=["clean", "attack"],
            output_dict=True,
            zero_division=0,
        ),
    }

    if len(np.unique(y_true)) == 2:
        result["roc_auc"] = float(roc_auc_score(y_true, score))
        result["pr_auc"] = float(average_precision_score(y_true, score))

    return result


def write_predictions(
    path: Path,
    metadata: pd.DataFrame,
    ids: np.ndarray,
    y_true: np.ndarray,
    score: np.ndarray,
    threshold: float,
) -> None:
    if len(metadata) == len(y_true):
        frame = metadata.copy()
    else:
        frame = pd.DataFrame({"id": ids})

    frame["true_flag"] = y_true
    frame["attack_probability"] = score
    frame["predicted_flag"] = (score >= threshold).astype(int)
    frame["decision"] = np.where(frame["predicted_flag"] == 1, "FLAG", "ALLOW")
    frame["is_correct"] = frame["true_flag"] == frame["predicted_flag"]
    frame.to_csv(path, index=False, encoding="utf-8-sig")


def top_feature_rows(
    model_name: str,
    model: Pipeline,
    feature_names: list[str],
    limit: int = 120,
) -> list[dict[str, Any]]:
    classifier = model.named_steps["classifier"]
    rows: list[dict[str, Any]] = []

    if model_name == "logistic_regression":
        values = classifier.coef_[0]
        for name, value in zip(feature_names, values):
            rows.append(
                {
                    "feature": name,
                    "importance": float(value),
                    "absolute_importance": abs(float(value)),
                    "direction": "toward_attack" if value > 0 else "toward_clean",
                }
            )
    elif model_name == "random_forest":
        for name, value in zip(feature_names, classifier.feature_importances_):
            rows.append(
                {
                    "feature": name,
                    "importance": float(value),
                    "absolute_importance": float(value),
                    "direction": "non_directional",
                }
            )

    return sorted(
        rows,
        key=lambda row: row["absolute_importance"],
        reverse=True,
    )[:limit]


def save_feature_importance(
    path: Path,
    rows: list[dict[str, Any]],
) -> None:
    if not rows:
        return
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def logistic_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    cv_folds: int,
    search_iterations: int,
    n_jobs: int,
) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
            (
                "classifier",
                LogisticRegression(
                    solver="saga",
                    class_weight="balanced",
                    max_iter=10000,
                    random_state=seed,
                    n_jobs=1,
                ),
            ),
        ]
    )

    parameter_distributions = [
        {
            "classifier__penalty": ["l2"],
            "classifier__C": [
                0.005, 0.01, 0.03, 0.05, 0.1, 0.3, 0.5,
                1.0, 2.0, 5.0, 10.0, 30.0,
            ],
        },
        {
            "classifier__penalty": ["elasticnet"],
            "classifier__C": [
                0.01, 0.03, 0.05, 0.1, 0.3, 0.5,
                1.0, 2.0, 5.0, 10.0,
            ],
            "classifier__l1_ratio": [0.1, 0.25, 0.5, 0.75, 0.9],
        },
    ]

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=search_iterations,
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=n_jobs,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    oof_score = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=n_jobs,
    )[:, 1]

    details = {
        "best_params": search.best_params_,
        "best_cross_validated_pr_auc": float(search.best_score_),
        "search_iterations": int(search_iterations),
        "cv_folds": int(cv_folds),
    }
    return best_model, oof_score, details


def random_forest_search(
    X_train: np.ndarray,
    y_train: np.ndarray,
    seed: int,
    cv_folds: int,
    search_iterations: int,
    n_jobs: int,
    trees: int,
) -> tuple[Pipeline, np.ndarray, dict[str, Any]]:
    pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            (
                "classifier",
                RandomForestClassifier(
                    n_estimators=trees,
                    class_weight="balanced_subsample",
                    bootstrap=True,
                    n_jobs=n_jobs,
                    random_state=seed,
                ),
            ),
        ]
    )

    parameter_distributions = {
        "classifier__criterion": ["gini", "entropy", "log_loss"],
        "classifier__max_depth": [None, 16, 24, 32, 48, 64],
        "classifier__min_samples_split": [2, 4, 6, 8, 12],
        "classifier__min_samples_leaf": [1, 2, 3, 4],
        "classifier__max_features": ["sqrt", "log2", 0.15, 0.25, 0.35, 0.5],
        "classifier__max_samples": [None, 0.70, 0.80, 0.90, 0.95],
        "classifier__class_weight": ["balanced", "balanced_subsample"],
    }

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    search = RandomizedSearchCV(
        estimator=pipeline,
        param_distributions=parameter_distributions,
        n_iter=search_iterations,
        scoring="average_precision",
        cv=cv,
        refit=True,
        random_state=seed,
        n_jobs=1,
        verbose=1,
        return_train_score=True,
    )
    search.fit(X_train, y_train)

    best_model = search.best_estimator_
    oof_score = cross_val_predict(
        clone(best_model),
        X_train,
        y_train,
        cv=cv,
        method="predict_proba",
        n_jobs=1,
    )[:, 1]

    classifier = best_model.named_steps["classifier"]
    details = {
        "best_params": search.best_params_,
        "best_cross_validated_pr_auc": float(search.best_score_),
        "search_iterations": int(search_iterations),
        "cv_folds": int(cv_folds),
        "trees_per_forest": int(classifier.n_estimators),
    }
    return best_model, oof_score, details


class ResidualBlock(nn.Module):
    def __init__(self, width: int, dropout: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(width, width),
            nn.BatchNorm1d(width),
            nn.Dropout(dropout),
        )
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class DeepGuardMLP(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 512,
        dropout: float = 0.35,
        residual_blocks: int = 2,
    ) -> None:
        super().__init__()

        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.residual_stack = nn.Sequential(
            *[
                ResidualBlock(hidden_dim, dropout)
                for _ in range(residual_blocks)
            ]
        )

        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, 256),
            nn.BatchNorm1d(256),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.GELU(),
            nn.Dropout(max(0.15, dropout - 0.10)),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(max(0.10, dropout - 0.15)),
            nn.Linear(64, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.input_projection(x)
        x = self.residual_stack(x)
        return self.classifier(x).squeeze(1)


def make_loader(
    X: np.ndarray,
    y: np.ndarray,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    dataset = TensorDataset(
        torch.from_numpy(X.astype(np.float32)),
        torch.from_numpy(y.astype(np.float32)),
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.inference_mode()
def torch_predict(
    model: nn.Module,
    X: np.ndarray,
    device: torch.device,
    batch_size: int,
) -> np.ndarray:
    model.eval()
    loader = DataLoader(
        TensorDataset(torch.from_numpy(X.astype(np.float32))),
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available(),
    )

    probabilities = []
    for (features,) in loader:
        features = features.to(device, non_blocking=True)
        logits = model(features)
        probabilities.append(torch.sigmoid(logits).cpu().numpy())

    return np.concatenate(probabilities)


def train_one_mlp_fold(
    X_fit: np.ndarray,
    y_fit: np.ndarray,
    X_valid: np.ndarray,
    y_valid: np.ndarray,
    X_test: np.ndarray,
    seed: int,
    device: torch.device,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    residual_blocks: int,
) -> tuple[nn.Module, np.ndarray, np.ndarray, dict[str, Any]]:
    seed_everything(seed)

    model = DeepGuardMLP(
        input_dim=X_fit.shape[1],
        hidden_dim=hidden_dim,
        dropout=dropout,
        residual_blocks=residual_blocks,
    ).to(device)

    positive = max(1, int((y_fit == 1).sum()))
    negative = max(1, int((y_fit == 0).sum()))
    positive_weight = torch.tensor(
        [negative / positive],
        dtype=torch.float32,
        device=device,
    )

    criterion = nn.BCEWithLogitsLoss(pos_weight=positive_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
        weight_decay=weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=8,
        min_lr=1e-6,
    )

    train_loader = make_loader(
        X_fit,
        y_fit,
        batch_size=batch_size,
        shuffle=True,
    )

    best_state = copy.deepcopy(model.state_dict())
    best_pr_auc = -math.inf
    best_epoch = 0
    stale_epochs = 0
    history = []

    for epoch in range(1, max_epochs + 1):
        model.train()
        epoch_losses = []

        for features, labels in train_loader:
            features = features.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            logits = model(features)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            epoch_losses.append(float(loss.detach().cpu()))

        valid_score = torch_predict(
            model,
            X_valid,
            device=device,
            batch_size=batch_size * 2,
        )
        valid_pr_auc = float(average_precision_score(y_valid, valid_score))
        scheduler.step(valid_pr_auc)

        history.append(
            {
                "epoch": epoch,
                "train_loss": float(np.mean(epoch_losses)),
                "validation_pr_auc": valid_pr_auc,
                "learning_rate": float(optimizer.param_groups[0]["lr"]),
            }
        )

        if valid_pr_auc > best_pr_auc + 1e-5:
            best_pr_auc = valid_pr_auc
            best_epoch = epoch
            best_state = copy.deepcopy(model.state_dict())
            stale_epochs = 0
        else:
            stale_epochs += 1

        if stale_epochs >= patience:
            break

    model.load_state_dict(best_state)

    valid_score = torch_predict(
        model,
        X_valid,
        device=device,
        batch_size=batch_size * 2,
    )
    test_score = torch_predict(
        model,
        X_test,
        device=device,
        batch_size=batch_size * 2,
    )

    details = {
        "best_epoch": int(best_epoch),
        "best_validation_pr_auc": float(best_pr_auc),
        "epochs_completed": int(len(history)),
        "final_learning_rate": float(optimizer.param_groups[0]["lr"]),
        "history": history,
    }
    return model, valid_score, test_score, details


def deep_mlp_cross_validated(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    output_dir: Path,
    seed: int,
    cv_folds: int,
    batch_size: int,
    max_epochs: int,
    patience: int,
    learning_rate: float,
    weight_decay: float,
    hidden_dim: int,
    dropout: float,
    residual_blocks: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Deep MLP device: {device}")

    cv = StratifiedKFold(
        n_splits=cv_folds,
        shuffle=True,
        random_state=seed,
    )

    oof_score = np.zeros(len(y_train), dtype=np.float32)
    test_fold_scores = []
    fold_details = []

    for fold_index, (fit_index, valid_index) in enumerate(
        cv.split(X_train, y_train),
        start=1,
    ):
        print(f"\nTraining deep MLP fold {fold_index}/{cv_folds}")

        imputer = SimpleImputer(strategy="median")
        scaler = StandardScaler()

        X_fit = imputer.fit_transform(X_train[fit_index])
        X_valid = imputer.transform(X_train[valid_index])
        X_test_fold = imputer.transform(X_test)

        X_fit = scaler.fit_transform(X_fit).astype(np.float32)
        X_valid = scaler.transform(X_valid).astype(np.float32)
        X_test_fold = scaler.transform(X_test_fold).astype(np.float32)

        fold_seed = seed + fold_index * 100
        model, valid_score, test_score, details = train_one_mlp_fold(
            X_fit=X_fit,
            y_fit=y_train[fit_index],
            X_valid=X_valid,
            y_valid=y_train[valid_index],
            X_test=X_test_fold,
            seed=fold_seed,
            device=device,
            batch_size=batch_size,
            max_epochs=max_epochs,
            patience=patience,
            learning_rate=learning_rate,
            weight_decay=weight_decay,
            hidden_dim=hidden_dim,
            dropout=dropout,
            residual_blocks=residual_blocks,
        )

        oof_score[valid_index] = valid_score
        test_fold_scores.append(test_score)

        preprocessor_path = output_dir / f"deep_mlp_preprocessor_fold_{fold_index}.joblib"
        joblib.dump(
            {"imputer": imputer, "scaler": scaler},
            preprocessor_path,
        )

        model_path = output_dir / f"deep_mlp_fold_{fold_index}.pt"
        torch.save(
            {
                "state_dict": model.state_dict(),
                "input_dim": int(X_train.shape[1]),
                "hidden_dim": int(hidden_dim),
                "dropout": float(dropout),
                "residual_blocks": int(residual_blocks),
                "fold": int(fold_index),
                "preprocessor": preprocessor_path.name,
            },
            model_path,
        )

        details["fold"] = fold_index
        details["model_path"] = model_path.name
        details["preprocessor_path"] = preprocessor_path.name
        fold_details.append(details)

    test_score = np.mean(np.vstack(test_fold_scores), axis=0)

    model_manifest = {
        "device_used_for_training": str(device),
        "cv_folds": cv_folds,
        "input_dim": int(X_train.shape[1]),
        "hidden_dim": int(hidden_dim),
        "dropout": float(dropout),
        "residual_blocks": int(residual_blocks),
        "batch_size": int(batch_size),
        "max_epochs": int(max_epochs),
        "patience": int(patience),
        "learning_rate": float(learning_rate),
        "weight_decay": float(weight_decay),
        "folds": fold_details,
    }
    return oof_score, test_score, model_manifest


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Advanced LR, Random Forest, and deep MLP training."
    )
    parser.add_argument("--feature_dir", default="guard_hard_features")
    parser.add_argument("--output_dir", default="guard_advanced_models")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--cv_folds", type=int, default=5)
    parser.add_argument("--minimum_recall", type=float, default=None)
    parser.add_argument("--n_jobs", type=int, default=-1)

    parser.add_argument("--lr_search_iterations", type=int, default=18)

    parser.add_argument("--rf_trees", type=int, default=2000)
    parser.add_argument("--rf_search_iterations", type=int, default=8)

    parser.add_argument("--mlp_hidden_dim", type=int, default=512)
    parser.add_argument("--mlp_residual_blocks", type=int, default=2)
    parser.add_argument("--mlp_dropout", type=float, default=0.35)
    parser.add_argument("--mlp_batch_size", type=int, default=32)
    parser.add_argument("--mlp_epochs", type=int, default=300)
    parser.add_argument("--mlp_patience", type=int, default=35)
    parser.add_argument("--mlp_learning_rate", type=float, default=3e-4)
    parser.add_argument("--mlp_weight_decay", type=float, default=1e-4)

    args = parser.parse_args()
    seed_everything(args.seed)

    feature_dir = Path(args.feature_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    X_train, y_train, train_ids = load_npz(feature_dir / "train_features.npz")
    X_test, y_test, test_ids = load_npz(feature_dir / "test_features.npz")

    train_metadata_path = feature_dir / "train_metadata.csv"
    test_metadata_path = feature_dir / "test_metadata.csv"
    train_metadata = (
        pd.read_csv(train_metadata_path)
        if train_metadata_path.exists()
        else pd.DataFrame({"id": train_ids})
    )
    test_metadata = (
        pd.read_csv(test_metadata_path)
        if test_metadata_path.exists()
        else pd.DataFrame({"id": test_ids})
    )

    manifest = json.loads(
        (feature_dir / "feature_names.json").read_text(encoding="utf-8")
    )
    feature_names = manifest["feature_names"]
    if X_train.shape[1] != len(feature_names):
        raise ValueError("Feature count does not match feature_names.json.")

    warnings.filterwarnings("always", category=ConvergenceWarning)

    complete_report: dict[str, Any] = {
        "train_shape": list(X_train.shape),
        "test_shape": list(X_test.shape),
        "feature_count": int(X_train.shape[1]),
        "models": {},
        "selection_policy": (
            "Hyperparameters and thresholds use training-set cross-validation "
            "or out-of-fold predictions. The fixed test set is used only once "
            "for final reporting."
        ),
    }
    comparison_rows = []

    # Logistic Regression
    print("\n=== Logistic Regression hyperparameter search ===")
    logistic_model, logistic_oof, logistic_details = logistic_search(
        X_train=X_train,
        y_train=y_train,
        seed=args.seed,
        cv_folds=args.cv_folds,
        search_iterations=args.lr_search_iterations,
        n_jobs=args.n_jobs,
    )
    logistic_threshold, logistic_threshold_details = select_threshold(
        y_train,
        logistic_oof,
        minimum_recall=args.minimum_recall,
    )
    logistic_model.fit(X_train, y_train)
    logistic_test_score = logistic_model.predict_proba(X_test)[:, 1]
    logistic_test_metrics = evaluate(
        y_test,
        logistic_test_score,
        logistic_threshold,
    )

    joblib.dump(
        logistic_model,
        output_dir / "logistic_regression_advanced.joblib",
    )
    write_predictions(
        output_dir / "logistic_regression_test_predictions.csv",
        test_metadata,
        test_ids,
        y_test,
        logistic_test_score,
        logistic_threshold,
    )
    save_feature_importance(
        output_dir / "logistic_regression_top_features.csv",
        top_feature_rows(
            "logistic_regression",
            logistic_model,
            feature_names,
        ),
    )

    complete_report["models"]["logistic_regression"] = {
        "training": logistic_details,
        "threshold_selection": logistic_threshold_details,
        "test": logistic_test_metrics,
    }
    comparison_rows.append(
        {
            "model": "logistic_regression",
            "threshold": logistic_threshold,
            "test_accuracy": logistic_test_metrics["accuracy"],
            "test_precision": logistic_test_metrics["precision"],
            "test_recall": logistic_test_metrics["recall"],
            "test_f1": logistic_test_metrics["f1"],
            "test_roc_auc": logistic_test_metrics["roc_auc"],
            "test_pr_auc": logistic_test_metrics["pr_auc"],
            "test_fpr": logistic_test_metrics["false_positive_rate"],
            "test_fnr": logistic_test_metrics["false_negative_rate"],
        }
    )

    # Random Forest
    print("\n=== Random Forest hyperparameter search ===")
    forest_model, forest_oof, forest_details = random_forest_search(
        X_train=X_train,
        y_train=y_train,
        seed=args.seed,
        cv_folds=args.cv_folds,
        search_iterations=args.rf_search_iterations,
        n_jobs=args.n_jobs,
        trees=args.rf_trees,
    )
    forest_threshold, forest_threshold_details = select_threshold(
        y_train,
        forest_oof,
        minimum_recall=args.minimum_recall,
    )
    forest_model.fit(X_train, y_train)
    forest_test_score = forest_model.predict_proba(X_test)[:, 1]
    forest_test_metrics = evaluate(
        y_test,
        forest_test_score,
        forest_threshold,
    )

    joblib.dump(
        forest_model,
        output_dir / "random_forest_advanced.joblib",
    )
    write_predictions(
        output_dir / "random_forest_test_predictions.csv",
        test_metadata,
        test_ids,
        y_test,
        forest_test_score,
        forest_threshold,
    )
    save_feature_importance(
        output_dir / "random_forest_top_features.csv",
        top_feature_rows(
            "random_forest",
            forest_model,
            feature_names,
        ),
    )

    complete_report["models"]["random_forest"] = {
        "training": forest_details,
        "threshold_selection": forest_threshold_details,
        "test": forest_test_metrics,
    }
    comparison_rows.append(
        {
            "model": "random_forest",
            "threshold": forest_threshold,
            "test_accuracy": forest_test_metrics["accuracy"],
            "test_precision": forest_test_metrics["precision"],
            "test_recall": forest_test_metrics["recall"],
            "test_f1": forest_test_metrics["f1"],
            "test_roc_auc": forest_test_metrics["roc_auc"],
            "test_pr_auc": forest_test_metrics["pr_auc"],
            "test_fpr": forest_test_metrics["false_positive_rate"],
            "test_fnr": forest_test_metrics["false_negative_rate"],
        }
    )

    # Deep MLP
    print("\n=== Deep MLP cross-validated ensemble ===")
    mlp_oof, mlp_test_score, mlp_manifest = deep_mlp_cross_validated(
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        output_dir=output_dir,
        seed=args.seed,
        cv_folds=args.cv_folds,
        batch_size=args.mlp_batch_size,
        max_epochs=args.mlp_epochs,
        patience=args.mlp_patience,
        learning_rate=args.mlp_learning_rate,
        weight_decay=args.mlp_weight_decay,
        hidden_dim=args.mlp_hidden_dim,
        dropout=args.mlp_dropout,
        residual_blocks=args.mlp_residual_blocks,
    )
    mlp_threshold, mlp_threshold_details = select_threshold(
        y_train,
        mlp_oof,
        minimum_recall=args.minimum_recall,
    )
    mlp_test_metrics = evaluate(
        y_test,
        mlp_test_score,
        mlp_threshold,
    )

    write_predictions(
        output_dir / "deep_mlp_test_predictions.csv",
        test_metadata,
        test_ids,
        y_test,
        mlp_test_score,
        mlp_threshold,
    )
    (output_dir / "deep_mlp_manifest.json").write_text(
        json.dumps(mlp_manifest, indent=2),
        encoding="utf-8",
    )

    complete_report["models"]["deep_mlp"] = {
        "training": mlp_manifest,
        "threshold_selection": mlp_threshold_details,
        "test": mlp_test_metrics,
    }
    comparison_rows.append(
        {
            "model": "deep_mlp",
            "threshold": mlp_threshold,
            "test_accuracy": mlp_test_metrics["accuracy"],
            "test_precision": mlp_test_metrics["precision"],
            "test_recall": mlp_test_metrics["recall"],
            "test_f1": mlp_test_metrics["f1"],
            "test_roc_auc": mlp_test_metrics["roc_auc"],
            "test_pr_auc": mlp_test_metrics["pr_auc"],
            "test_fpr": mlp_test_metrics["false_positive_rate"],
            "test_fnr": mlp_test_metrics["false_negative_rate"],
        }
    )

    comparison = pd.DataFrame(comparison_rows).sort_values(
        ["test_f1", "test_recall", "test_precision"],
        ascending=False,
    )
    comparison.to_csv(
        output_dir / "model_comparison.csv",
        index=False,
        encoding="utf-8-sig",
    )

    complete_report["final_test_results"] = comparison.to_dict(orient="records")
    (output_dir / "metrics.json").write_text(
        json.dumps(complete_report, indent=2),
        encoding="utf-8",
    )

    model_manifest = {
        "label_meaning": {"0": "clean/allow", "1": "attack/flag"},
        "feature_manifest": str(
            (feature_dir / "feature_names.json").resolve()
        ),
        "models": {
            "logistic_regression": "logistic_regression_advanced.joblib",
            "random_forest": "random_forest_advanced.joblib",
            "deep_mlp": "deep_mlp_manifest.json",
        },
        "thresholds": {
            "logistic_regression": logistic_threshold,
            "random_forest": forest_threshold,
            "deep_mlp": mlp_threshold,
        },
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(model_manifest, indent=2),
        encoding="utf-8",
    )

    print("\n=== Final model comparison ===")
    print(comparison.to_string(index=False))
    print(f"\nSaved outputs to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()
