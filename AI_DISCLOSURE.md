# Use of AI

**Tools used:** Claude, ChatGPT


**How they were used:**
1. Generating the boilerplate code for the FastAPI service `partA/eta_app.py`, and understanding how pydantic `Field` constraints and a `model_validator` enforce the ranges and the cross-field (infeasible) rule, how an HTTP middleware can attach `X-Request-ID` to every response, how a custom validation handler shapes the 422 body, and how the Prometheus text format represents a counter and a histogram (`_bucket`, `+Inf`, `_sum`, `_count`)
1. Generating the pytest suites `partA/test_eta_app.py` and `partB/test_gate.py`, including the use of `monkeypatch` to force a 500 and boundary cases just inside and just outside each gate threshold
1. Generating the boilerplate code for `partB/eta_mlflow.py`: the custom pyfunc wrapper, logging with a signature, input example and pinned requirements, registering the models with aliases, the pure `decide()` function, the promotion gate and the shadow comparison. Also understanding why moving an alias does not change a running `mlflow models serve` process
1. Generating the boilerplate code for the Ray Serve application `partC/eta_ray.py` and the verification script `partC/verify_ray.py`, and understanding why a stable hash (SHA-256) is needed instead of Python's salted `hash()`, how fractional CPUs keep the application within 2 CPUs, and how to choose the autoscaling parameters
1. Generating the smoke test `partD/smoke_eta.py` and brainstorming the model-sanity checks and the four canary abort criteria

**Impact:**

LLMs greatly reduced the time I would have otherwise spent writing boilerplate for four different serving stacks, which let me spend more time running the experiments myself and understanding the reasoning behind them, such as saturation, promotion gates, alias resolution and sticky canary routing.
