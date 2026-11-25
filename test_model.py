"""
Validation checks for model.py (SENG 331 - P03).

Run with:   python test_model.py
(Also works with pytest if it is installed.)
"""

import copy

import model
from model import DEFAULT_PARAMS, EVENT_NAMES, STATIONS, fake_result, run_simulation

TOP_KEYS = {"timeseries", "customers", "event_log", "end_time"}
CUSTOMER_KEYS = {"cid", "type", "arrive", "order_start", "order_end", "fulfill_start", "fulfill_end"}
EVENT_KEYS = {"t", "cid", "type", "event", "server"}


def capacities(params):
    return {"order": params["n_order_takers"],
            "walkin_fulfill": params["n_walkin_workers"],
            "drive_fulfill": params["n_drive_workers"]}


def station_of(event):
    """The station whose server an event starts or ends, or None for arrive."""
    if event["event"] in ("order_start", "order_end"):
        return "order"
    if event["event"] in ("fulfill_start", "fulfill_end"):
        return f"{event['type']}_fulfill"
    return None


def check_result(res, params, label):
    """Run every structural and consistency check on one result dict."""
    cap = capacities(params)

    # ---- Exact schema ----
    assert set(res) == TOP_KEYS, label
    ts = res["timeseries"]
    assert set(ts) == {"Q", "B", "L"}, label
    assert set(ts["Q"]) == set(STATIONS) and set(ts["B"]) == set(STATIONS), label
    assert isinstance(res["end_time"], float), label
    series = {("Q", s): ts["Q"][s] for s in STATIONS}
    series.update({("B", s): ts["B"][s] for s in STATIONS})
    series[("L",)] = ts["L"]
    for key, pts in series.items():
        assert isinstance(pts, list) and pts[0] == (0.0, 0), (label, key)
        for pt in pts:
            assert isinstance(pt, tuple) and len(pt) == 2, (label, key, pt)
        times = [t for t, _ in pts]
        assert times == sorted(times), (label, key, "timeseries not chronological")
        values = [v for _, v in pts]
        assert min(values) >= 0, (label, key, "negative value")
        assert values[-1] == 0, (label, key, "system not empty at the end")
        if key[0] == "B":
            assert max(values) <= cap[key[1]], (label, key, "busy exceeds capacity")

    customers = res["customers"]
    log = res["event_log"]
    for c in customers:
        assert set(c) == CUSTOMER_KEYS, (label, c)
        assert c["type"] in ("walkin", "drive"), (label, c)
    for e in log:
        assert set(e) == EVENT_KEYS, (label, e)
        assert e["event"] in EVENT_NAMES, (label, e)
        if e["event"] in ("order_start", "fulfill_start"):
            assert isinstance(e["server"], int) and e["server"] >= 0, (label, e)
        else:
            assert e["server"] is None, (label, e)

    # ---- Customers: unique IDs, complete and ordered timestamps ----
    cids = [c["cid"] for c in customers]
    assert len(cids) == len(set(cids)), (label, "duplicate cid")
    for c in customers:
        stamps = [c[k] for k in ("arrive", "order_start", "order_end", "fulfill_start", "fulfill_end")]
        assert all(isinstance(x, float) for x in stamps), (label, "incomplete record", c)
        assert stamps == sorted(stamps), (label, "timestamps out of order", c)
        assert c["arrive"] < params["sim_time"], (label, "arrival at/after cutoff", c)

    # ---- end_time ----
    expected_end = max((c["fulfill_end"] for c in customers), default=0.0)
    assert res["end_time"] == expected_end, label
    if log:
        assert log[-1]["t"] == res["end_time"], label

    # ---- Event log: chronological, 5 events per customer, matches records ----
    times = [e["t"] for e in log]
    assert times == sorted(times), (label, "event log not chronological")
    assert len(log) == 5 * len(customers), label
    by_cid = {}
    for e in log:
        by_cid.setdefault(e["cid"], []).append(e)
    recs = {c["cid"]: c for c in customers}
    for cid, evs in by_cid.items():
        # The order in the log must be the causal lifecycle order (this matters for ties).
        assert [e["event"] for e in evs] == list(EVENT_NAMES), (label, cid)
        for e in evs:
            assert e["t"] == recs[cid][e["event"]] and e["type"] == recs[cid]["type"], (label, e)

    # ---- Servers: never hold two customers at once, IDs within capacity ----
    holder = {s: {} for s in STATIONS}          # station -> server id -> cid
    server_of = {}                              # (cid, station) -> server id
    for e in log:
        st = station_of(e)
        if e["event"].endswith("_start"):
            sid = e["server"]
            assert sid < cap[st], (label, "server id out of range", e)
            assert sid not in holder[st], (label, "server double-assigned", e)
            holder[st][sid] = e["cid"]
            server_of[(e["cid"], st)] = sid
        elif e["event"].endswith("_end"):
            sid = server_of[(e["cid"], st)]
            assert holder[st].pop(sid) == e["cid"], (label, e)
    assert all(not h for h in holder.values()), label

    # ---- Event log and timeseries agree: replay the log on its own ----
    q = {s: 0 for s in STATIONS}
    b = {s: 0 for s in STATIONS}
    n = 0
    expected = {k: [(0.0, 0)] for k in series}
    for e in log:
        t, ev, f = e["t"], e["event"], f"{e['type']}_fulfill"
        if ev == "arrive":
            n += 1; q["order"] += 1
            expected[("L",)].append((t, n)); expected[("Q", "order")].append((t, q["order"]))
        elif ev == "order_start":
            q["order"] -= 1; b["order"] += 1
            expected[("Q", "order")].append((t, q["order"])); expected[("B", "order")].append((t, b["order"]))
        elif ev == "order_end":
            b["order"] -= 1; q[f] += 1
            expected[("B", "order")].append((t, b["order"])); expected[("Q", f)].append((t, q[f]))
        elif ev == "fulfill_start":
            q[f] -= 1; b[f] += 1
            expected[("Q", f)].append((t, q[f])); expected[("B", f)].append((t, b[f]))
        elif ev == "fulfill_end":
            b[f] -= 1; n -= 1
            expected[("B", f)].append((t, b[f])); expected[("L",)].append((t, n))
    for key in series:
        assert series[key] == expected[key], (label, key, "timeseries disagrees with event log")

    # FCFS check on the shared order queue: order service starts in arrival order.
    order_starts = [e["cid"] for e in log if e["event"] == "order_start"]
    arrivals = [e["cid"] for e in log if e["event"] == "arrive"]
    assert order_starts == arrivals, (label, "order queue not FCFS")


def max_wait(res):
    return max((max(c["order_start"] - c["arrive"], c["fulfill_start"] - c["order_end"])
                for c in res["customers"]), default=0.0)


# ---------------------------------------------------------------------------

def test_default_run():
    res = run_simulation(DEFAULT_PARAMS, seed=0)
    check_result(res, DEFAULT_PARAMS, "default seed=0")
    assert res["end_time"] >= max(c["arrive"] for c in res["customers"])
    types = {c["type"] for c in res["customers"]}
    assert types == {"walkin", "drive"}


def test_many_seeds():
    for seed in range(20):
        check_result(run_simulation(DEFAULT_PARAMS, seed=seed), DEFAULT_PARAMS, f"seed={seed}")


def test_reproducible():
    assert run_simulation(DEFAULT_PARAMS, 42) == run_simulation(DEFAULT_PARAMS, 42)
    assert run_simulation(DEFAULT_PARAMS, 42) != run_simulation(DEFAULT_PARAMS, 43)
    assert fake_result(7) == fake_result(7)
    assert fake_result(7) != fake_result(8)


def test_params_not_mutated():
    before = copy.deepcopy(DEFAULT_PARAMS)
    custom = copy.deepcopy(DEFAULT_PARAMS)
    custom["n_order_takers"] = 4
    custom_before = copy.deepcopy(custom)
    run_simulation(DEFAULT_PARAMS, 1)
    run_simulation(custom, 1)
    fake_result(1)
    assert DEFAULT_PARAMS == before and custom == custom_before


def test_order_takers_2_and_4():
    for n in (2, 4):
        params = copy.deepcopy(DEFAULT_PARAMS)
        params["n_order_takers"] = n
        for seed in range(5):
            res = run_simulation(params, seed)
            check_result(res, params, f"n_order_takers={n} seed={seed}")
        # With n=4, server ID 3 should actually be used at some point.
        used = {e["server"] for e in res["event_log"] if e["event"] == "order_start"}
        assert used == set(range(n)), (n, used)


def test_short_horizon_and_no_customers():
    params = copy.deepcopy(DEFAULT_PARAMS)
    params["sim_time"] = 30
    res = run_simulation(params, 3)
    check_result(res, params, "sim_time=30")
    assert res["end_time"] > 0
    # So small a horizon that no arrival is possible: documented convention end_time == 0.0.
    params["sim_time"] = 1e-9
    res = run_simulation(params, 0)
    check_result(res, params, "no customers")
    assert res["customers"] == [] and res["event_log"] == [] and res["end_time"] == 0.0


def test_heavy_load_ties():
    # Deterministic service times create many simultaneous events; checks tie handling.
    params = copy.deepcopy(DEFAULT_PARAMS)
    params.update(sim_time=120, walkin_mean_iat=1.0, drive_mean_iat=1.0,
                  order_time={"low": 1, "mode": 1, "high": 1.0000001},
                  fulfill_time={"mean": 4, "sd": 0})
    for seed in range(5):
        res = run_simulation(params, seed)
        check_result(res, params, f"heavy seed={seed}")
        assert max_wait(res) > 0


def test_fake_result():
    for seed in range(20):
        res = fake_result(seed)
        check_result(res, DEFAULT_PARAMS, f"fake seed={seed}")
        assert {c["type"] for c in res["customers"]} == {"walkin", "drive"}
        assert max(c["order_start"] - c["arrive"] for c in res["customers"]) > 0
    # fake_result must not depend on SimPy
    assert "run_simulation" not in model.fake_result.__code__.co_names


def test_same_schema():
    real, fake = run_simulation(DEFAULT_PARAMS, 0), fake_result(0)
    assert set(real) == set(fake)
    assert set(real["timeseries"]) == set(fake["timeseries"])
    for k in ("Q", "B"):
        assert set(real["timeseries"][k]) == set(fake["timeseries"][k])


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    res = run_simulation()
    print(f"\nAll {len(tests)} checks passed.")
    print(f"Default run (seed=0): {len(res['customers'])} customers, end_time = {res['end_time']:.2f} min")
