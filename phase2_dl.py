"""
phase2_dl.py
Phase 2 — Multi-class attack classification (DoS / Probe / R2L / U2R).
Uses sklearn MLPClassifier (fast, no TF overhead, cacheable with joblib).
TensorFlow path available via USE_TF = True if you want GPU training.
"""

import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

USE_TF = False   # flip to True only if you have a GPU and want TF

if USE_TF:
    try:
        import os; os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
        import tensorflow as tf
        from tensorflow import keras
        TF_AVAILABLE = True
    except ImportError:
        TF_AVAILABLE = False
else:
    TF_AVAILABLE = False

ANOMALY_CATS = ["DoS", "Probe", "R2L", "U2R"]


class Phase2DeepANN:
    def __init__(self, epochs=10, batch_size=64, test_size=0.3, random_state=42):
        self.epochs       = epochs
        self.batch_size   = batch_size
        self.test_size    = test_size
        self.random_state = random_state
        self.optimizer    = "Adam"
        self.loss_fn      = "Sparse Categorical CE"

        self.model               = None
        self.accuracy            = 0.0
        self.report              = {}
        self.attack_distribution = {}
        self.attack_counts       = {}
        self.architecture        = []

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self, loader, phase1=None):
        X_tr, X_te, y_tr, y_te = loader.get_train_test_multiclass(
            test_size=self.test_size, random_state=self.random_state
        )
        n_features = X_tr.shape[1]
        n_classes  = len(ANOMALY_CATS)

        # Subsample to 20k for speed
        MAX_TRAIN = 20_000
        if len(X_tr) > MAX_TRAIN:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        print(f"[Phase2] Training ANN on {len(X_tr)} anomaly samples "
              f"({'TensorFlow' if TF_AVAILABLE else 'sklearn MLP'})…")

        if TF_AVAILABLE:
            self.model = self._build_keras(n_features, n_classes)
            self.model.fit(X_tr, y_tr, epochs=self.epochs,
                           batch_size=self.batch_size,
                           validation_split=0.1, verbose=0)
            y_pred = np.argmax(self.model.predict(X_te, verbose=0), axis=1)
        else:
            self.model = MLPClassifier(
                hidden_layer_sizes=(64, 32),
                activation="relu",
                solver="adam",
                batch_size=self.batch_size,
                max_iter=30,
                early_stopping=True,
                n_iter_no_change=5,
                random_state=self.random_state,
                verbose=False,
            )
            self.model.fit(X_tr, y_tr)
            y_pred = self.model.predict(X_te)

        self.accuracy = accuracy_score(y_te, y_pred)

        report_raw  = classification_report(
            y_te, y_pred, target_names=ANOMALY_CATS,
            output_dict=True, zero_division=0
        )
        self.report = self._format_report(report_raw)

        # Attack distribution from full anomaly set
        mask   = loader.y_binary == 1
        y_full = loader.y_multiclass[mask] - 1
        total  = len(y_full)
        for i, cat in enumerate(ANOMALY_CATS):
            cnt = int((y_full == i).sum())
            self.attack_counts[cat]       = cnt
            self.attack_distribution[cat] = round(cnt / total * 100, 1) if total else 0

        self.architecture = [
            {"name": "INPUT",  "size": n_features, "activation": None},
            {"name": "DENSE",  "size": 64,         "activation": "ReLU"},
            {"name": "DENSE",  "size": 32,         "activation": "ReLU"},
            {"name": "OUTPUT", "size": n_classes,  "activation": "Softmax"},
        ]

        print(f"[Phase2] Done. Accuracy: {self.accuracy:.6f} | "
              f"Dist: {self.attack_distribution}")
        return self

    @staticmethod
    def _build_keras(n_features, n_classes):
        model = keras.Sequential([
            keras.layers.Input(shape=(n_features,)),
            keras.layers.Dense(64, activation="relu"),
            keras.layers.Dropout(0.3),
            keras.layers.Dense(32, activation="relu"),
            keras.layers.Dropout(0.2),
            keras.layers.Dense(n_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam",
                      loss="sparse_categorical_crossentropy",
                      metrics=["accuracy"])
        return model

    @staticmethod
    def _format_report(raw):
        out = {}
        for cat in ANOMALY_CATS:
            if cat in raw:
                out[cat] = {
                    "precision": round(raw[cat]["precision"], 4),
                    "recall":    round(raw[cat]["recall"],    4),
                    "f1":        round(raw[cat]["f1-score"],  4),
                    "support":   int(raw[cat]["support"]),
                }
        return out