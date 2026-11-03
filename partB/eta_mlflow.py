import argparse
import json
import os
import sys
import time
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import eta_data
import mlflow
import mlflow.pyfunc
from mlflow import MlflowClient
from mlflow.models import infer_signature
os.environ.setdefault("MLFLOW_DISABLE_TELEMETRY", "true")
mlflow.set_tracking_uri(os.getenv("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))

NAME = "eta-model"
EXPERIMENT = "eta-serving"
FEATURES = eta_data.FEATURES
RUSH_HOURS = [7, 8, 9, 16, 17, 18, 19]
DATA = ROOT / "data"

# pyfunc, registration
class EtaPyfunc(mlflow.pyfunc.PythonModel): # type: ignore
    def load_context(self, context):
        b = joblib.load(context.artifacts["bundle"])
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]

    def predict(self, context, model_input, params=None): # type: ignore
        X = pd.DataFrame(model_input)[self.cols]
        eta = np.round(self.model.predict(X).astype(float), 1)
        return pd.DataFrame({"eta_minutes": eta, "model_version": [self.version] * len(X)})

def _pinned_requirements():
    import cloudpickle, sklearn
    return [f"mlflow=={mlflow.__version__}", f"scikit-learn=={sklearn.__version__}", f"pandas=={pd.__version__}", f"numpy=={np.__version__}", f"joblib=={joblib.__version__}", f"cloudpickle=={cloudpickle.__version__}"]

def register():
    from sklearn.metrics import mean_absolute_error
    mlflow.set_experiment(EXPERIMENT)
    client = MlflowClient()
    train = pd.read_csv(DATA / "eta_train.csv")
    Xf = train[FEATURES].astype("float64")
    out_example = pd.DataFrame({"eta_minutes": [42.0], "model_version": ["v1"]})
    signature = infer_signature(Xf.head(5), out_example)
    reqs = _pinned_requirements()

    registered = {}
    for ver in ("v1", "v2"):
        path = DATA / f"eta_{ver}.joblib"
        b = joblib.load(path)
        mae = mean_absolute_error(train.eta_minutes, b["model"].predict(train[FEATURES]))
        with mlflow.start_run(run_name=f"eta-{ver}"):
            mlflow.log_params({"algorithm": type(b["model"]).__name__, "bundle_version": ver})
            mlflow.log_metric("train_mae", mae)
            info = mlflow.pyfunc.log_model(name="model", python_model=EtaPyfunc(), artifacts={"bundle": str(path)},
                signature=signature, input_example=Xf.head(2), pip_requirements=reqs, registered_model_name=NAME)
        registered[ver] = info.registered_model_version
        print(f"{ver}: {type(b['model']).__name__:28s} train MAE {mae:.2f} -> registry version " f"{info.registered_model_version}")

    client.set_registered_model_alias(NAME, "prod", registered["v1"])
    client.set_registered_model_alias(NAME, "candidate", registered["v2"])
    print("\npinned pip_requirements:", reqs)
    print_registry_table(client)

def print_registry_table(client=None):
    client = client or MlflowClient()
    aliases = client.get_registered_model(NAME).aliases
    rows = []
    for v in sorted(client.search_model_versions(f"name='{NAME}'"), key=lambda v: int(v.version)):
        bundle_ver = client.get_run(v.run_id).data.params.get("bundle_version", "?") # type: ignore
        rows.append({"version": v.version, "aliases": ",".join(a for a, n in aliases.items() if str(n) == str(v.version)) or "-", "bundle": bundle_ver, "run_id": v.run_id[:8]}) # type: ignore
    print("\nRegistry table for", NAME)
    print(pd.DataFrame(rows).to_string(index=False))

# promotion gate
def decide(prod, cand, min_rel_gain=0.03, max_p95_ms=10.0, max_slice_regression=0.05):
    eps = 1e-12
    rel_gain = (prod["mae"] - cand["mae"]) / prod["mae"] if prod["mae"] > 0 else 0.0
    criteria = {
        "mae_rel_gain": {"value": round(rel_gain, 4), "threshold": f">= {min_rel_gain}",
                         "pass": bool(rel_gain >= min_rel_gain - eps)},
        "p95_ms": {"value": round(cand["p95_ms"], 3), "threshold": f"<= {max_p95_ms}",
                   "pass": bool(cand["p95_ms"] <= max_p95_ms + eps)},
    }
    for s in ("mae_rush", "mae_rain"):
        limit = prod[s] * (1 + max_slice_regression)
        criteria[f"{s}_no_regression"] = {"value": round(cand[s], 3), "threshold": f"<= {round(limit, 3)}",
                                          "pass": bool(cand[s] <= limit + eps)}
    return {"criteria": criteria, "promote": all(c["pass"] for c in criteria.values())}

def evaluate(alias, fresh, n_latency=200):
    m = mlflow.pyfunc.load_model(f"models:/{NAME}@{alias}")
    X = fresh[FEATURES].astype("float64")
    pred = m.predict(X)["eta_minutes"].to_numpy()
    err = np.abs(pred - fresh["eta_minutes"].to_numpy())
    rush = fresh["hour"].isin(RUSH_HOURS).to_numpy()
    rain = (fresh["rain"] == 1).to_numpy()
    for i in range(10):
        m.predict(X.iloc[[i]])
    lat = []
    for i in range(n_latency):
        row = X.iloc[[i]]
        t0 = time.perf_counter()
        m.predict(row)
        lat.append((time.perf_counter() - t0) * 1000)
    version = m.predict(X.iloc[[0]])["model_version"].iloc[0]
    return {"model_version": version, "mae": float(err.mean()), "mae_rush": float(err[rush].mean()), "mae_rain": float(err[rain].mean()), "p95_ms": float(np.percentile(lat, 95))}

def promotion_gate(**thresholds):
    fresh = eta_data.make_data(2000, seed=99)
    prod, cand = evaluate("prod", fresh), evaluate("candidate", fresh)
    result = decide(prod, cand, **thresholds)
    return {"thresholds": thresholds or "defaults", "prod": _r(prod), "candidate": _r(cand), **result}

def _r(d):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in d.items()}

# shadow comparison
def shadow(prod_url, cand_url, n=200):
    import requests
    rows = pd.read_csv(DATA / "eta_fresh.csv").head(n)
    truth = rows["eta_minutes"].to_numpy()
    recs = rows[FEATURES].to_dict(orient="records")
    s = requests.Session()
    out = {"prod": {"pred": [], "lat": [], "ver": set()}, "candidate": {"pred": [], "lat": [], "ver": set()}}

    def call(arm, url, rec):
        t0 = time.perf_counter()
        r = s.post(url, json={"dataframe_records": [rec]}, timeout=10)
        out[arm]["lat"].append((time.perf_counter() - t0) * 1000)
        r.raise_for_status()
        p = r.json()["predictions"][0]
        out[arm]["pred"].append(p["eta_minutes"])
        out[arm]["ver"].add(p["model_version"])

    for arm, url in (("prod", prod_url), ("candidate", cand_url)):
        for rec in recs[:5]:
            s.post(url, json={"dataframe_records": [rec]}, timeout=10) # type: ignore
    for i, rec in enumerate(recs):
        order = [("prod", prod_url), ("candidate", cand_url)]
        for arm, url in (order if i % 2 == 0 else order[::-1]):
            call(arm, url, rec)

    p, c = np.array(out["prod"]["pred"]), np.array(out["candidate"]["pred"])
    diff = np.abs(p - c)
    res = {
        "n_requests_each": n,
        "versions": {"prod": sorted(out["prod"]["ver"]), "candidate": sorted(out["candidate"]["ver"])},
        "mean_abs_diff_min": round(float(diff.mean()), 2),
        "pct_diff_gt_10_min": round(float((diff > 10).mean() * 100), 1),
        "p95_ms": {"prod": round(float(np.percentile(out["prod"]["lat"], 95)), 2),
                   "candidate": round(float(np.percentile(out["candidate"]["lat"], 95)), 2)},
        "p50_ms": {"prod": round(float(np.percentile(out["prod"]["lat"], 50)), 2),
                   "candidate": round(float(np.percentile(out["candidate"]["lat"], 50)), 2)},
        "mae_vs_truth_on_these_200": {"prod": round(float(np.abs(p - truth).mean()), 2),
                                      "candidate": round(float(np.abs(c - truth).mean()), 2)},
    }
    return res

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("register")
    sub.add_parser("table")
    g = sub.add_parser("gate")
    g.add_argument("--min-rel-gain", type=float)
    g.add_argument("--max-p95-ms", type=float)
    g.add_argument("--max-slice-regression", type=float)
    sh = sub.add_parser("shadow")
    sh.add_argument("--prod-url", default="http://127.0.0.1:5101/invocations")
    sh.add_argument("--cand-url", default="http://127.0.0.1:5102/invocations")
    sh.add_argument("-n", type=int, default=200)
    a = ap.parse_args()

    if a.cmd == "register":
        register()
    elif a.cmd == "table":
        print_registry_table()
    elif a.cmd == "gate":
        th = {k: v for k, v in {"min_rel_gain": a.min_rel_gain, "max_p95_ms": a.max_p95_ms, "max_slice_regression": a.max_slice_regression}.items() if v is not None}
        print(json.dumps(promotion_gate(**th), indent=2))
    elif a.cmd == "shadow":
        print(json.dumps(shadow(a.prod_url, a.cand_url, a.n), indent=2))