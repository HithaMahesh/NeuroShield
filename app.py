"""
app.py — Flask backend for NeuroShield
- Trains all models at startup in a background thread
- Saves trained models to disk (models/ folder)
- On next run, loads from disk in ~3 seconds instead of re-training
- Flask reloader disabled to prevent TF restart loops
"""

import os, threading, traceback, joblib, json
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

# ── Globals ───────────────────────────────────────────────────────────────
_loader = _phase1 = _phase2 = _phase3 = None
_ready  = False
_status = "Initializing…"
_error  = None

MODEL_DIR   = "models"           # folder where trained models are saved
CACHE_META  = os.path.join(MODEL_DIR, "meta.json")   # accuracy / stats cache
CACHE_P1    = os.path.join(MODEL_DIR, "phase1_rf.pkl")
CACHE_P2    = os.path.join(MODEL_DIR, "phase2_ann.pkl")
CACHE_P3    = os.path.join(MODEL_DIR, "phase3_rl.pkl")
CACHE_LOADER= os.path.join(MODEL_DIR, "loader_stats.pkl")

# ── Check if a valid cache exists ────────────────────────────────────────
def cache_exists():
    return all(os.path.exists(p) for p in
               [CACHE_P1, CACHE_P2, CACHE_P3, CACHE_LOADER, CACHE_META])

# ── Training / loading thread ─────────────────────────────────────────────
def train_all():
    global _loader, _phase1, _phase2, _phase3, _ready, _status, _error
    try:
        os.makedirs(MODEL_DIR, exist_ok=True)

        # ── LOAD DATASET (always needed for sample_events) ─────────────
        _status = "Loading dataset…"
        from nsl_kdd_loader import NSLKDDLoader
        _loader = NSLKDDLoader()
        _loader.load()

        if cache_exists():
            # ── FAST PATH: restore from disk ───────────────────────────
            _status = "Loading saved models from disk…"
            print("[App] Cache found — loading models from disk…")

            _phase1 = joblib.load(CACHE_P1)
            print("[App] Phase 1 loaded ✓")

            _phase2 = joblib.load(CACHE_P2)
            print("[App] Phase 2 loaded ✓")

            _phase3 = joblib.load(CACHE_P3)
            print("[App] Phase 3 loaded ✓")

        else:
            # ── SLOW PATH: train from scratch, then save ───────────────
            _status = "Training Phase 1 — Random Forest (30 trees)…"
            from phase1_anomaly_rf import Phase1RandomForest
            _phase1 = Phase1RandomForest(n_estimators=30)
            _phase1.train(_loader)
            joblib.dump(_phase1, CACHE_P1, compress=3)
            print("[App] Phase 1 saved to disk ✓")

            _status = "Training Phase 2 — Deep ANN…"
            from phase2_dl import Phase2DeepANN
            _phase2 = Phase2DeepANN()
            _phase2.train(_loader, _phase1)
            joblib.dump(_phase2, CACHE_P2, compress=3)
            print("[App] Phase 2 saved to disk ✓")

            _status = "Training Phase 3 — Q-Learning…"
            from phase3_rl import Phase3QLearning
            _phase3 = Phase3QLearning()
            _phase3.train(_phase1, _loader)
            joblib.dump(_phase3, CACHE_P3, compress=3)
            print("[App] Phase 3 saved to disk ✓")

            # Save metadata so we can verify cache matches dataset
            meta = {
                "total_records":   int(_loader.total_records),
                "num_features":    int(_loader.num_features),
                "phase1_accuracy": float(_phase1.accuracy),
                "phase2_accuracy": float(_phase2.accuracy),
            }
            with open(CACHE_META, "w") as f:
                json.dump(meta, f)

        _ready  = True
        _status = "All systems ready."
        print("[App] ✅ Ready at http://127.0.0.1:5000")

    except Exception as e:
        _error  = traceback.format_exc()
        _status = f"Error: {e}"
        print("[App] ❌ Error:\n", _error)


# ── Routes ────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/network")
def network_viz():
    return render_template("network.html")


@app.route("/api/status")
def api_status():
    return jsonify({"ready": _ready, "status": _status, "error": _error})


@app.route("/api/clear_cache", methods=["POST"])
def api_clear_cache():
    """Call this if you change your dataset and want to re-train."""
    import shutil
    if os.path.exists(MODEL_DIR):
        shutil.rmtree(MODEL_DIR)
    return jsonify({"ok": True, "message": "Cache cleared. Restart app.py to re-train."})


@app.route("/api/overview")
def api_overview():
    if not _ready:
        return jsonify({"error": _status}), 503
    try:
        return jsonify({
            "total_records":   int(_loader.total_records),
            "anomaly_records": int(_loader.anomaly_records),
            "normal_records":  int(_loader.normal_records),
            "num_features":    int(_loader.num_features),
            "phase1_accuracy": round(float(_phase1.accuracy), 6),
            "phase2_accuracy": round(float(_phase2.accuracy), 6),
            "rl_best_reward":  round(float(_phase3.best_reward), 2),
            "missing_values":  int(_loader.missing_values),
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/phase1")
def api_phase1():
    if not _ready:
        return jsonify({"error": _status}), 503
    try:
        return jsonify({
            "accuracy":      round(float(_phase1.accuracy), 6),
            "report":        _phase1.report,
            "cm":            _phase1.confusion_matrix,
            "n_estimators":  _phase1.n_estimators,
            "test_size":     _phase1.test_size,
            "n_features":    _phase1.n_features,
            "train_samples": _phase1.train_samples,
            "test_samples":  _phase1.test_samples,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/phase2")
def api_phase2():
    if not _ready:
        return jsonify({"error": _status}), 503
    try:
        return jsonify({
            "accuracy":      round(float(_phase2.accuracy), 6),
            "attack_dist":   _phase2.attack_distribution,
            "architecture":  _phase2.architecture,
            "optimizer":     _phase2.optimizer,
            "loss":          _phase2.loss_fn,
            "epochs":        _phase2.epochs,
            "batch_size":    _phase2.batch_size,
            "report":        _phase2.report,
            "attack_counts": _phase2.attack_counts,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/phase3")
def api_phase3():
    if not _ready:
        return jsonify({"error": _status}), 503
    try:
        return jsonify({
            "q_table":       _phase3.q_table_serializable(),
            "best_reward":   round(float(_phase3.best_reward), 2),
            "alpha":         _phase3.alpha,
            "gamma":         _phase3.gamma,
            "epsilon":       _phase3.epsilon,
            "episodes":      _phase3.episodes,
            "reward_matrix": _phase3.reward_matrix,
            "policy":        _phase3.policy,
            "n_states":      _phase3.n_states,
            "n_actions":     _phase3.n_actions,
            "n_samples":     _phase3.n_samples,
        })
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


@app.route("/api/simulate")
def api_simulate():
    if not _ready:
        return jsonify({"error": _status}), 503
    try:
        n      = int(request.args.get("n", 20))
        events = _loader.sample_events(n)
        results = []
        for ev in events:
            pred   = _phase1.predict_one(ev["features"])
            action = _phase3.act(pred, explore=True)
            results.append({
                "ip":       ev["src_ip"],
                "proto":    ev["protocol"],
                "service":  ev["service"],
                "bytes":    ev["bytes"],
                "status":   "ANOMALY" if pred == 1 else "NORMAL",
                "category": ev.get("attack_cat", "—"),
                "action":   action.upper(),
            })
        return jsonify(results)
    except Exception:
        return jsonify({"error": traceback.format_exc()}), 500


# ── Entry point ───────────────────────────────────────────────────────────
if __name__ == "__main__":
    t = threading.Thread(target=train_all, daemon=True)
    t.start()
    app.run(debug=False, port=5000, use_reloader=False, threaded=True)