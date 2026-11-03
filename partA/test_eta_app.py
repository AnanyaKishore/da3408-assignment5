import re
import uuid
import pandas as pd
import pytest
from fastapi.testclient import TestClient
import eta_app

URL = "/v1/models/eta:predict"
GOOD = {"distance_km": 12.5, "n_stops": 4, "hour": 8, "weekday": 2,"traffic_index": 0.4, "rain": 0, "weight_kg": 6.0}

@pytest.fixture
def client():
    return TestClient(eta_app.app)

def body(*rows):
    return {"instances": list(rows)}

def assert_validation_error(r):
    assert r.status_code == 422
    j = r.json()
    assert j["error"] == "validation_failed"
    assert isinstance(j["details"], list) and j["details"]
    assert {"field", "msg"} <= set(j["details"][0])
    assert j["request_id"] == r.headers["X-Request-ID"]
    return j

# valid requests
def test_valid_single_request(client):
    r = client.post(URL, json=body(GOOD))
    assert r.status_code == 200
    j = r.json()
    assert set(j) == {"predictions", "model_version", "request_id"}
    assert len(j["predictions"]) == 1
    p = j["predictions"][0]
    assert isinstance(p, float) and p == round(p, 1)
    assert j["model_version"] == eta_app.VERSION

def test_batch_order_preserved(client):
    rows = [{**GOOD, "distance_km": d} for d in (5.0, 80.0, 20.0, 140.0, 1.0)]
    j = client.post(URL, json=body(*rows)).json()
    expected = [round(float(p), 1) for p in eta_app.model.predict(pd.DataFrame(rows)[eta_app.FEATURES])]
    assert j["predictions"] == expected
    j_rev = client.post(URL, json=body(*rows[::-1])).json()
    assert j_rev["predictions"] == expected[::-1]

def test_healthz(client):
    r = client.get("/healthz")
    assert r.status_code == 200
    assert r.json() == {"status": "ok", "model_version": eta_app.VERSION}

# invalid inputs
def test_out_of_range_value_rejected(client):
    j = assert_validation_error(client.post(URL, json=body({**GOOD, "hour": 24})))
    assert any("hour" in d["field"] for d in j["details"])

def test_missing_field_rejected(client):
    row = {k: v for k, v in GOOD.items() if k != "weight_kg"}
    j = assert_validation_error(client.post(URL, json=body(row)))
    assert any("weight_kg" in d["field"] for d in j["details"])

def test_wrong_type_rejected(client):
    j = assert_validation_error(client.post(URL, json=body({**GOOD, "distance_km": "far"})))
    assert any("distance_km" in d["field"] for d in j["details"])

@pytest.mark.parametrize("field,value", [("rain", 2), ("weight_kg", 0), ("traffic_index", 1.01), ("distance_km", 0.05), ("n_stops", 61), ("weekday", 7)])
def test_range_edges_rejected(client, field, value):
    assert_validation_error(client.post(URL, json=body({**GOOD, field: value})))

# cross field rule
def test_infeasible_delivery_rejected(client):
    r = client.post(URL, json=body({**GOOD, "distance_km": 151, "n_stops": 21}))
    j = assert_validation_error(r)
    assert any("infeasible" in d["msg"] for d in j["details"])

def test_cross_field_boundary_is_feasible(client):
    # exactly 150 km / 20 stops is allowed (rule is strictly greater than)
    r = client.post(URL, json=body({**GOOD, "distance_km": 150, "n_stops": 20}))
    assert r.status_code == 200

# batch size limits
def test_empty_batch_rejected(client):
    assert_validation_error(client.post(URL, json=body()))

def test_batch_of_500_accepted_501_rejected(client):
    r = client.post(URL, json={"instances": [GOOD] * 500})
    assert r.status_code == 200 and len(r.json()["predictions"]) == 500
    assert_validation_error(client.post(URL, json={"instances": [GOOD] * 501}))

# request id
def test_request_id_echoed(client):
    r = client.post(URL, json=body(GOOD), headers={"X-Request-ID": "abc-123"})
    assert r.headers["X-Request-ID"] == "abc-123"
    assert r.json()["request_id"] == "abc-123"

def test_request_id_generated_when_absent(client):
    r = client.post(URL, json=body(GOOD))
    rid = r.headers["X-Request-ID"]
    uuid.UUID(rid)
    assert r.json()["request_id"] == rid

def test_request_id_on_error_responses(client):
    r = client.post(URL, json=body({**GOOD, "hour": -1}), headers={"X-Request-ID": "err-42"})
    assert r.status_code == 422
    assert r.headers["X-Request-ID"] == "err-42" and r.json()["request_id"] == "err-42"
    r2 = client.post(URL, json=body({**GOOD, "hour": -1}))
    uuid.UUID(r2.headers["X-Request-ID"])
    assert r2.json()["request_id"] == r2.headers["X-Request-ID"]

# metrics
def _counter(text, path, status):
    m = re.search(rf'eta_requests_total\{{path="{re.escape(path)}",status="{status}"\}} (\d+)', text)
    return int(m.group(1)) if m else 0

def test_metrics_output(client):
    before = client.get("/metrics").text
    client.post(URL, json=body(GOOD))
    client.post(URL, json=body({**GOOD, "hour": 99}))
    text = client.get("/metrics").text
    assert _counter(text, URL, 200) == _counter(before, URL, 200) + 1
    assert _counter(text, URL, 422) == _counter(before, URL, 422) + 1
    assert "# TYPE eta_request_latency_seconds histogram" in text
    assert len(re.findall(r'eta_request_latency_seconds_bucket\{le="[^"]+"\} \d+', text)) >= 3
    assert 'eta_request_latency_seconds_bucket{le="+Inf"}' in text
    assert re.search(r"^eta_request_latency_seconds_sum [\d.]+$", text, re.M)
    count = int(re.search(r"^eta_request_latency_seconds_count (\d+)$", text, re.M).group(1)) # type: ignore
    inf = int(re.search(r'le="\+Inf"\} (\d+)', text).group(1)) # type: ignore
    assert count == inf
    assert 'path="/metrics"' not in text

# generic 500
class Boom:
    def predict(self, X):
        raise RuntimeError("secret internal detail: /opt/models/xyz")

def test_inference_failure_returns_generic_500(client, monkeypatch):
    monkeypatch.setattr(eta_app, "model", Boom())
    r = client.post(URL, json=body(GOOD), headers={"X-Request-ID": "boom-1"})
    assert r.status_code == 500
    assert r.headers["X-Request-ID"] == "boom-1"
    j = r.json()
    assert j == {"error": "internal_error", "request_id": "boom-1"}
    assert "secret" not in r.text and "RuntimeError" not in r.text and "Traceback" not in r.text