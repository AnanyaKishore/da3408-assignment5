import argparse
import math
import sys
import time
import requests

BASE_ROW = {"distance_km": 30.0, "n_stops": 6, "hour": 11, "weekday": 2, "traffic_index": 0.3, "rain": 0, "weight_kg": 8.0}
PREDICT = "/v1/models/eta:predict"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", required=True)
    ap.add_argument("--expect-version", required=True)
    ap.add_argument("--max-p95-ms", type=float, default=80.0)
    ap.add_argument("--beta", action="store_true", help="send X-Beta: 1 (Ray router: force the candidate)")
    a = ap.parse_args()

    base = a.url.split(PREDICT)[0].rstrip("/")
    url = base + PREDICT
    s = requests.Session()
    if a.beta:
        s.headers["X-Beta"] = "1"
    results = []

    def check(name, fn):
        try:
            ok, info = fn()
        except Exception as e:
            ok, info = False, f"{type(e).__name__}: {e}"
        results.append((name, ok, info))
        print(f"[{'PASS' if ok else 'FAIL'}] {name}: {info}")

    def predict(rows):
        r = s.post(url, json={"customer_id": "smoke-test", "instances": rows}, timeout=10)
        r.raise_for_status()
        return r.json()

    def c_health():
        r = s.get(base + "/healthz", timeout=5)
        return r.status_code == 200 and r.json().get("status") == "ok", f"HTTP {r.status_code} {r.text[:80]}"

    def c_valid():
        rows = [BASE_ROW, {**BASE_ROW, "distance_km": 5.0}, {**BASE_ROW, "n_stops": 15}]
        j = predict(rows)
        p = j["predictions"]
        ok = (len(p) == len(rows) and all(isinstance(x, (int, float)) and math.isfinite(x) and 0 < x < 1000 for x in p)
              and j.get("model_version") == a.expect_version)
        return ok, f"predictions={p} model_version={j.get('model_version')} (expected {a.expect_version})"

    def c_distance():
        dists = [2.0, 10.0, 25.0, 50.0, 100.0]
        p = predict([{**BASE_ROW, "distance_km": d} for d in dists])["predictions"]
        ok = all(b >= a_ for a_, b in zip(p, p[1:]))
        return ok, f"distance {dists} -> eta {p}"

    def c_traffic():
        tr = [0.0, 0.25, 0.5, 0.75, 1.0]
        p = predict([{**BASE_ROW, "traffic_index": t} for t in tr])["predictions"]
        ok = all(b >= a_ for a_, b in zip(p, p[1:]))
        return ok, f"traffic {tr} -> eta {p}"

    def c_stops():
        st = [1, 5, 10, 20]
        p = predict([{**BASE_ROW, "n_stops": n} for n in st])["predictions"]
        ok = all(b >= a_ for a_, b in zip(p, p[1:]))
        return ok, f"n_stops {st} -> eta {p}"

    def c_invalid():
        codes = []
        for bad in ({**BASE_ROW, "hour": 25}, {**BASE_ROW, "distance_km": 200, "n_stops": 30}):
            r = s.post(url, json={"customer_id": "smoke-test", "instances": [bad]}, timeout=10)
            codes.append(r.status_code)
        return all(c == 422 for c in codes), f"status codes {codes} (expected [422, 422])"

    def c_latency():
        for _ in range(5):
            predict([BASE_ROW])
        lat = []
        for _ in range(50):
            t0 = time.perf_counter()
            predict([BASE_ROW])
            lat.append((time.perf_counter() - t0) * 1000)
        lat.sort()
        p95 = lat[min(len(lat) - 1, int(0.95 * len(lat)))]
        return p95 <= a.max_p95_ms, f"p95 = {p95:.1f} ms (limit {a.max_p95_ms} ms)"

    for name, fn in [("healthz", c_health), ("valid batch + version", c_valid), ("sanity: distance monotone", c_distance), ("sanity: traffic monotone", c_traffic), ("sanity: stops monotone", c_stops), ("invalid input -> 422", c_invalid), ("latency p95", c_latency)]:
        check(name, fn)
        
    failed = [n for n, ok, _ in results if not ok]
    print(f"\n{len(results) - len(failed)}/{len(results)} checks passed" + (f"; FAILED: {failed}" if failed else ""))
    sys.exit(1 if failed else 0)

if __name__ == "__main__":
    main()