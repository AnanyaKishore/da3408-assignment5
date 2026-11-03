import sys
import time
from pathlib import Path
import pandas as pd
import ray
import requests
from ray import serve
sys.path.insert(0, str(Path(__file__).resolve().parent))
import eta_ray

ROOT = Path(__file__).resolve().parent.parent
BASE = "http://127.0.0.1:8100"
URL = f"{BASE}/v1/models/eta:predict"
APP = "eta"
N = 1000
IDS = [f"customer-{i:05d}" for i in range(N)]
ROW = pd.read_csv(ROOT / "data" / "eta_fresh.csv")[eta_ray.FEATURES].iloc[0].to_dict()
sess = requests.Session()

def deploy(weight):
    serve.run(eta_ray.build(weight=weight), name=APP, route_prefix="/")
    t0 = time.time()
    while time.time() - t0 < 60:
        try:
            if sess.get(f"{BASE}/healthz", timeout=2).json().get("weight") == weight:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise TimeoutError("app did not come up")

def arm_of(cid, beta=False):
    r = sess.post(URL, json={"customer_id": cid, "instances": [ROW]}, headers={"X-Beta": "1"} if beta else {}, timeout=10) # type: ignore
    assert r.status_code == 200, (r.status_code, r.text)
    j = r.json()
    assert j["model_version"] == j["arm"], j
    return j["arm"]

def arms_for(ids):
    return {cid: arm_of(cid) for cid in ids}

def header(t):
    print(f"\n=== {t}")

ray.init(num_cpus=2, include_dashboard=False, ignore_reinit_error=True, logging_level="ERROR", log_to_driver=False) # type: ignore
serve.start(http_options={"host": "127.0.0.1", "port": 8100}, logging_config={"log_level": "WARNING"})
print("Ray cluster resources:", {k: v for k, v in ray.cluster_resources().items() if k in ("CPU", "memory")})
deploy(0.2)
st = serve.status().applications[APP]
print("Deployments:", {n: str(d.status).split(".")[-1] for n, d in st.deployments.items()})
print("CPU per replica: EtaModel 0.25, Router 0.1 -> worst case 0.1 + 2*0.25 + 3*0.25 = 1.35 CPUs")
print("Candidate autoscaling_config:", eta_ray.CANDIDATE_AUTOSCALING)

header("(a) share of 1000 customers routed to the candidate at weight 0.2")
first = arms_for(IDS)
share = sum(a == "v2" for a in first.values()) / N
print(f"v2: {sum(a == 'v2' for a in first.values())}  v1: {sum(a == 'v1' for a in first.values())}  "
      f"share = {share:.3f} (target 0.200 +- 0.040)")
assert abs(share - 0.2) <= 0.04, share
print("PASS (a)")

header("(b) 200 customers called a second time")
sample = IDS[::5]
second = arms_for(sample)
changed = [c for c in sample if second[c] != first[c]]
print(f"same arm: {len(sample) - len(changed)}/200  changed: {len(changed)}  "
      f"(of these 200, {sum(first[c] == 'v2' for c in sample)} are on v2)")
assert not changed, changed[:5]
print("PASS (b)")

header("(c) redeploy the identical app (delete and run) and call all 1000 again")
serve.delete(APP)
time.sleep(2)
deploy(0.2)
after = arms_for(IDS)
changed = [c for c in IDS if after[c] != first[c]]
print(f"same arm after redeploy: {N - len(changed)}/{N}  changed: {len(changed)}")
assert not changed, changed[:5]
print("PASS (c)")

header("(d) raise weight 0.2 -> 0.5")
deploy(0.5)
raised = arms_for(IDS)
moved_back = [c for c in IDS if first[c] == "v2" and raised[c] == "v1"]
moved_up = [c for c in IDS if first[c] == "v1" and raised[c] == "v2"]
share5 = sum(a == "v2" for a in raised.values()) / N
print(f"share now {share5:.3f}; v1->v2: {len(moved_up)}; v2->v1: {len(moved_back)}; "
      f"stayed on v2: {sum(first[c] == 'v2' and raised[c] == 'v2' for c in IDS)}")
assert not moved_back, moved_back[:5]
print("PASS (d)")

header("(e) X-Beta: 1 on 50 customers that are normally on prod (v1)")
prod_customers = [c for c in IDS if raised[c] == "v1"][:50]
forced = [arm_of(c, beta=True) for c in prod_customers]
without = [arm_of(c) for c in prod_customers]
print(f"with header -> v2: {forced.count('v2')}/50;  same customers without header -> v1: {without.count('v1')}/50")
assert len(prod_customers) == 50 and all(a == "v2" for a in forced) and all(a == "v1" for a in without)
print("PASS (e)")

st = serve.status().applications[APP]
print("\nReplica states at the end:", {n: dict(d.replica_states) for n, d in st.deployments.items()})
print("\nALL CHECKS PASSED")
serve.shutdown()
ray.shutdown()