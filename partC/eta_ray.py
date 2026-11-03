import hashlib
import os
import time
from pathlib import Path
from typing import Dict, List, Optional
import joblib
import pandas as pd
from fastapi import FastAPI, Header
from pydantic import BaseModel, Field, model_validator
from ray import serve
from ray.serve.handle import DeploymentHandle

ROOT = Path(__file__).resolve().parent.parent
FEATURES = ["distance_km", "n_stops", "hour", "weekday", "traffic_index", "rain", "weight_kg"]
CANDIDATE_AUTOSCALING = {"min_replicas": 1, "max_replicas": 3, "initial_replicas": 1, "target_ongoing_requests": 2, "upscale_delay_s": 5, "downscale_delay_s": 60, "look_back_period_s": 15}

# stable bucketing
def bucket(customer_id: str) -> float:
    h = hashlib.sha256(customer_id.encode("utf-8")).digest()
    return int.from_bytes(h[:8], "big") / 2 ** 64

def choose_arm(customer_id: str, weight: float) -> str:
    return "v2" if bucket(customer_id) < weight else "v1"

# model deployment
@serve.deployment(ray_actor_options={"num_cpus": 0.25}, max_ongoing_requests=16, health_check_period_s=5, logging_config={"log_level": "WARNING"})
class EtaModel:
    def __init__(self, path: str, extra_work_ms: float = 0.0):
        b = joblib.load(path)
        self.model, self.cols, self.version = b["model"], b["feature_columns"], b["version"]
        self.extra = extra_work_ms / 1000.0
        self.replica = serve.get_replica_context().replica_id.unique_id

    async def __call__(self, rows: List[Dict]):
        X = pd.DataFrame(rows)[self.cols]
        preds = self.model.predict(X)
        if self.extra:
            time.sleep(self.extra)
        return {"predictions": [round(float(p), 1) for p in preds], "model_version": self.version, "served_by": self.replica}

    async def check_health(self):
        if self.model is None:
            raise RuntimeError("model missing")
        
# ingress router
class Delivery(BaseModel):
    distance_km: float = Field(ge=0.1, le=500)
    n_stops: int = Field(ge=1, le=60)
    hour: int = Field(ge=0, le=23)
    weekday: int = Field(ge=0, le=6)
    traffic_index: float = Field(ge=0, le=1)
    rain: int = Field(ge=0, le=1)
    weight_kg: float = Field(gt=0, le=1000)

    @model_validator(mode="after")
    def feasible(self):
        if self.distance_km > 150 and self.n_stops > 20:
            raise ValueError("infeasible delivery: distance_km > 150 and n_stops > 20")
        return self

class RoutedRequest(BaseModel):
    customer_id: str = Field(min_length=1)
    instances: List[Delivery] = Field(min_length=1, max_length=500)

api = FastAPI(title="SwiftParcel ETA (Ray Serve, sticky canary)")
@serve.deployment(num_replicas=1, ray_actor_options={"num_cpus": 0.1}, logging_config={"log_level": "WARNING"})
@serve.ingress(api)
class Router:
    def __init__(self, prod: DeploymentHandle, cand: DeploymentHandle, weight: float):
        self.handles = {"v1": prod, "v2": cand}
        self.weight = float(weight)

    @api.get("/healthz")
    def healthz(self):
        return {"status": "ok", "weight": self.weight}

    @api.post("/v1/models/eta:predict")
    async def predict(self, req: RoutedRequest, x_beta: Optional[str] = Header(default=None)):
        arm = "v2" if (x_beta or "").strip() == "1" else choose_arm(req.customer_id, self.weight)
        out = await self.handles[arm].remote([d.model_dump() for d in req.instances])
        return {**out, "arm": arm, "customer_id": req.customer_id}

def build(weight: float = 0.1, extra_work_ms: float = 0.0):
    v1_path, v2_path = str(ROOT / "data" / "eta_v1.joblib"), str(ROOT / "data" / "eta_v2.joblib")
    prod = EtaModel.options(name="EtaModelV1", num_replicas=2).bind(v1_path, extra_work_ms)
    cand = EtaModel.options(name="EtaModelV2", autoscaling_config=CANDIDATE_AUTOSCALING).bind(v2_path, extra_work_ms)
    return Router.bind(prod, cand, weight)

app = build(float(os.getenv("CANARY_WEIGHT", "0.1")))
if __name__ == "__main__":
    import argparse
    import ray
    ap = argparse.ArgumentParser()
    ap.add_argument("--weight", type=float, default=0.1)
    ap.add_argument("--port", type=int, default=8100)
    a = ap.parse_args()
    ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True)
    serve.start(http_options={"host": "127.0.0.1", "port": a.port})
    serve.run(build(weight=a.weight), name="eta", route_prefix="/")
    print(f"Ray Serve ETA app up on http://127.0.0.1:{a.port}  (weight={a.weight}). Ctrl+C to stop.")
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        serve.shutdown()
        ray.shutdown()