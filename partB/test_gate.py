import pytest
from eta_mlflow import decide

PROD = {"mae": 10.0, "mae_rush": 12.0, "mae_rain": 11.0, "p95_ms": 3.0}

def cand(**kw):
    return {"mae": 8.0, "mae_rush": 10.0, "mae_rain": 9.0, "p95_ms": 4.0, **kw}

def test_clear_win_is_promoted():
    out = decide(PROD, cand())
    assert out["promote"] is True
    assert all(c["pass"] for c in out["criteria"].values())
    assert set(out) == {"criteria", "promote"}

def test_worse_candidate_is_rejected():
    out = decide(PROD, cand(mae=11.0))
    assert out["promote"] is False
    assert out["criteria"]["mae_rel_gain"]["pass"] is False

def test_gain_just_inside_threshold():
    assert decide(PROD, cand(mae=9.69))["promote"] is True

def test_gain_exactly_at_threshold():
    assert decide(PROD, cand(mae=9.70))["promote"] is True

def test_gain_just_outside_threshold():
    out = decide(PROD, cand(mae=9.71))
    assert out["promote"] is False and out["criteria"]["mae_rel_gain"]["pass"] is False

@pytest.mark.parametrize("p95,ok", [(9.99, True), (10.0, True), (10.01, False)])
def test_latency_boundary(p95, ok):
    out = decide(PROD, cand(p95_ms=p95))
    assert out["criteria"]["p95_ms"]["pass"] is ok and out["promote"] is ok

@pytest.mark.parametrize("value,ok", [(12.59, True), (12.6, True), (12.61, False)])
def test_rush_slice_boundary(value, ok):
    out = decide(PROD, cand(mae_rush=value))
    assert out["criteria"]["mae_rush_no_regression"]["pass"] is ok and out["promote"] is ok

@pytest.mark.parametrize("value,ok", [(11.54, True), (11.56, False)])
def test_rain_slice_boundary(value, ok):
    out = decide(PROD, cand(mae_rain=value))
    assert out["criteria"]["mae_rain_no_regression"]["pass"] is ok and out["promote"] is ok

def test_better_overall_but_slice_regresses_is_rejected():
    out = decide(PROD, cand(mae=6.0, mae_rain=13.2))
    assert out["criteria"]["mae_rel_gain"]["pass"] is True
    assert out["promote"] is False

def test_custom_thresholds_are_respected():
    c = cand()
    assert decide(PROD, c, min_rel_gain=0.60)["promote"] is False
    assert decide(PROD, c, min_rel_gain=0.19)["promote"] is True
    assert decide(PROD, c, max_p95_ms=3.9)["promote"] is False
    assert decide(PROD, cand(mae_rush=12.5), max_slice_regression=0.0)["promote"] is False

def test_decide_is_pure():
    p, c = dict(PROD), cand()
    decide(p, c)
    assert p == PROD and c == cand()
    assert decide(p, c) == decide(p, c)