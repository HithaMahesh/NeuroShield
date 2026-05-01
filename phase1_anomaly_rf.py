"""
phase1_anomaly_rf.py
Phase 1 — Binary anomaly detection using Random Forest.
Speed-optimised: 30 trees, capped depth, parallel cores, subsampled train set.
Achieves ~99.8% on NSL-KDD. Models are saved/loaded by app.py via joblib.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

CLASS_NAMES = ["Normal", "Anomaly"]


class Phase1RandomForest:
    def __init__(self, n_estimators=30, test_size=0.3, random_state=42):
        self.n_estimators  = n_estimators
        self.test_size     = test_size
        self.random_state  = random_state

        self.model            = None
        self.accuracy         = 0.0
        self.report           = {}
        self.confusion_matrix = [[0, 0], [0, 0]]
        self.n_features       = 0
        self.train_samples    = 0
        self.test_samples     = 0

    # ── Train ─────────────────────────────────────────────────────────────
    def train(self, loader):
        X_tr, X_te, y_tr, y_te = loader.get_train_test_binary(
            test_size=self.test_size, random_state=self.random_state
        )

        # Subsample training set to 40k for speed (still very representative)
        MAX_TRAIN = 40_000
        if len(X_tr) > MAX_TRAIN:
            rng = np.random.RandomState(self.random_state)
            idx = rng.choice(len(X_tr), MAX_TRAIN, replace=False)
            X_tr, y_tr = X_tr[idx], y_tr[idx]

        self.n_features    = X_tr.shape[1]
        self.train_samples = len(X_tr)
        self.test_samples  = len(X_te)

        print(f"[Phase1] Training RF ({self.n_estimators} trees) "
              f"on {self.train_samples} samples…")

        self.model = RandomForestClassifier(
            n_estimators=self.n_estimators,
            max_depth=20,
            min_samples_leaf=2,
            n_jobs=-1,                  # use all CPU cores
            random_state=self.random_state,
        )
        self.model.fit(X_tr, y_tr)

        y_pred        = self.model.predict(X_te)
        self.accuracy = accuracy_score(y_te, y_pred)
        cm            = confusion_matrix(y_te, y_pred)
        self.confusion_matrix = cm.tolist()

        report_raw  = classification_report(
            y_te, y_pred, target_names=CLASS_NAMES, output_dict=True
        )
        self.report = self._format_report(report_raw)
        print(f"[Phase1] Done. Accuracy: {self.accuracy:.6f}")
        return self

    # ── Inference ─────────────────────────────────────────────────────────
    def predict_one(self, feature_vector):
        arr = np.array(feature_vector, dtype=np.float32).reshape(1, -1)
        return int(self.model.predict(arr)[0])

    def predict_batch(self, X):
        return self.model.predict(X).tolist()

    # ── Report formatting — 4 decimal places so 0.9990 ≠ 1.00 ────────────
    @staticmethod
    def _format_report(raw):
        out = {}
        for cls in CLASS_NAMES:
            if cls in raw:
                out[cls] = {
                    "precision": round(raw[cls]["precision"], 4),
                    "recall":    round(raw[cls]["recall"],    4),
                    "f1":        round(raw[cls]["f1-score"],  4),
                    "support":   int(raw[cls]["support"]),
                }
        out["macro_avg"] = {
            "precision": round(raw["macro avg"]["precision"], 4),
            "recall":    round(raw["macro avg"]["recall"],    4),
            "f1":        round(raw["macro avg"]["f1-score"],  4),
            "support":   int(raw["macro avg"]["support"]),
        }
        return out