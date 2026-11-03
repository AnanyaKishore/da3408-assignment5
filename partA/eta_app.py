import os
import time
import uuid
from pathlib import Path
from typing import List
import joblib
import pandas as pd
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, model_validator

# model, loaded once per process
def _resolve(path: str) -> str:
    p = Path(path)
    if p.exists() or p.is_absolute():
        return str(p)
    alt = Path(__file__).resolve().parent.parent / p
    return str(alt) if alt.exists() else str(p)

MODEL_PATH = _resolve(os.getenv("ETA_MODEL", "data/eta_v1.joblib"))
bundle = joblib.load(MODEL_PATH)
model = bundle["model"]
FEATURES = bundle["feature_columns"]
VERSION = bundle["version"]

# request schema
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

class PredictRequest(BaseModel):
    instances: List[Delivery] = Field(min_length=1, max_length=500)

app = FastAPI(title="SwiftParcel ETA service")

# Prometheus metrics
BUCKETS = (0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5)
REQ_COUNT = {} # (path, status): count
LAT_BUCKETS = [0] * len(BUCKETS)
LAT = {"sum": 0.0, "count": 0}

def _observe(path: str, status: int, seconds: float):
    key = (path, str(status))
    REQ_COUNT[key] = REQ_COUNT.get(key, 0) + 1
    for i, le in enumerate(BUCKETS):
        if seconds <= le:
            LAT_BUCKETS[i] += 1
            break
    LAT["sum"] += seconds
    LAT["count"] += 1

def _render_metrics() -> str:
    lines = ["# HELP eta_requests_total Total HTTP requests by path and status.","# TYPE eta_requests_total counter"]
    for (path, status), n in sorted(REQ_COUNT.items()):
        lines.append(f'eta_requests_total{{path="{path}",status="{status}"}} {n}')
    lines += ["# HELP eta_request_latency_seconds Request latency in seconds.", "# TYPE eta_request_latency_seconds histogram"]
    cum = 0
    for le, n in zip(BUCKETS, LAT_BUCKETS):
        cum += n
        lines.append(f'eta_request_latency_seconds_bucket{{le="{le}"}} {cum}')
    lines.append(f'eta_request_latency_seconds_bucket{{le="+Inf"}} {LAT["count"]}')
    lines.append(f'eta_request_latency_seconds_sum {LAT["sum"]:.6f}')
    lines.append(f'eta_request_latency_seconds_count {LAT["count"]}')
    return "\n".join(lines) + "\n"

# request id, metrics
@app.middleware("http")
async def request_id_and_metrics(request: Request, call_next):
    rid = request.headers.get("x-request-id") or str(uuid.uuid4())
    request.state.request_id = rid
    t0 = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception: # generic 500, no internals
        response = JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    response.headers["X-Request-ID"] = rid
    if request.url.path != "/metrics":
        _observe(request.url.path, response.status_code, time.perf_counter() - t0)
    return response

@app.exception_handler(RequestValidationError)
async def on_validation_error(request: Request, exc: RequestValidationError):
    details = []
    for e in exc.errors():
        loc = [str(p) for p in e.get("loc", ()) if p != "body"]
        details.append({"field": ".".join(loc) or "body", "msg": e.get("msg", "invalid")})
    return JSONResponse(status_code=422, content={"error": "validation_failed", "details": details,"request_id": request.state.request_id})

# endpoints
@app.post("/v1/models/eta:predict")
def predict(req: PredictRequest, request: Request):
    rid = request.state.request_id
    try:
        X = pd.DataFrame([d.model_dump() for d in req.instances])[FEATURES]
        preds = model.predict(X)
        out = [round(float(p), 1) for p in preds]
    except Exception: # don't leak to client
        return JSONResponse(status_code=500, content={"error": "internal_error", "request_id": rid})
    return {"predictions": out, "model_version": VERSION, "request_id": rid}

@app.get("/healthz")
def healthz():
    return {"status": "ok", "model_version": VERSION}

@app.get("/metrics")
def metrics():
    return Response(content=_render_metrics(), media_type="text/plain; version=0.0.4")