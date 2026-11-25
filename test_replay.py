"""
Validation checks for replay.py (SENG 331 - P03).

Run with:   python test_replay.py
(Also works with pytest if it is installed.)
"""

import bisect
import copy

from model import DEFAULT_PARAMS, fake_result, run_simulation
from replay import STATIONS, state_at

RESULT_KEYS = {"t", "queues", "servers", "types", "stats"}


def ev(t, cid, ctype, event, server=None):
    return {"t": t, "cid": cid, "type": ctype, "event": event, "server": server}


def value_at(series, t):
    """Value of a (time, value) step series right after time t (last point with time <= t)."""
    times = [p[0] for p in series]
    return series[bisect.bisect_right(times, t) - 1][1]


def check_shape(state, params):
    assert set(state) == RESULT_KEYS
    assert set(state["queues"]) == set(STATIONS) and set(state["servers"]) == set(STATIONS)
    assert set(state["stats"]) == {"served", "in_system"}
    assert len(state["servers"]["order"]) == params["n_order_takers"]
    assert len(state["servers"]["walkin_fulfill"]) == params["n_walkin_workers"]
    assert len(state["servers"]["drive_fulfill"]) == params["n_drive_workers"]


def check_against_timeseries(res, params, times):
    """At every given time, replay must match Q, B and L from the model, and
    every customer must be in the right place for its type."""
    ts = res["timeseries"]
    for t in times:
        s = state_at(res["event_log"], t, params)
        check_shape(s, params)
        for st in STATIONS:
            assert len(s["queues"][st]) == value_at(ts["Q"][st], t), (t, st, "Q")
            busy = sum(c is not None for c in s["servers"][st])
            assert busy == value_at(ts["B"][st], t), (t, st, "B")
        assert s["stats"]["in_system"] == value_at(ts["L"], t), (t, "L")

        # Every customer in the system appears exactly once, at the right kind of station.
        placed = []
        for st in STATIONS:
            placed += s["queues"][st] + [c for c in s["servers"][st] if c is not None]
            if st != "order":
                for c in s["queues"][st] + [c for c in s["servers"][st] if c is not None]:
                    assert st == f"{s['types'][c]}_fulfill", (t, c, st)
        assert sorted(placed) == sorted(s["types"]), t
        departed = sum(1 for e in res["event_log"] if e["event"] == "fulfill_end" and e["t"] <= t)
        assert s["stats"]["served"] == departed, t


def sample_times(res):
    """Every event time, the midpoints between them, plus before-start and after-end."""
    ev_times = sorted({e["t"] for e in res["event_log"]})
    mids = [(a + b) / 2 for a, b in zip(ev_times, ev_times[1:])]
    return [-1.0, 0.0] + ev_times + mids + [res["end_time"] + 10]


# ---------------------------------------------------------------------------

def test_initial_empty_state():
    for log in ([], fake_result(0)["event_log"]):
        s = state_at(log, 0.0, DEFAULT_PARAMS)
        check_shape(s, DEFAULT_PARAMS)
        assert s == {
            "t": 0.0,
            "queues": {"order": [], "walkin_fulfill": [], "drive_fulfill": []},
            "servers": {"order": [None] * 3, "walkin_fulfill": [None] * 2, "drive_fulfill": [None] * 2},
            "types": {},
            "stats": {"served": 0, "in_system": 0},
        }
    assert state_at([], 1e9, DEFAULT_PARAMS)["stats"] == {"served": 0, "in_system": 0}


def test_hand_built_lifecycle():
    """A small log written by hand, checked step by step: queue transitions,
    server IDs, FIFO order, both types, and simultaneous events."""
    log = [
        ev(1.0, 0, "walkin", "arrive"),
        ev(1.0, 0, "walkin", "order_start", 0),
        ev(1.5, 1, "drive", "arrive"),
        ev(1.5, 1, "drive", "order_start", 1),
        ev(2.0, 2, "walkin", "arrive"),          # waits: only 2 order takers below
        ev(2.5, 3, "drive", "arrive"),           # waits behind cid 2 (FIFO)
        ev(3.0, 0, "walkin", "order_end"),
        ev(3.0, 2, "walkin", "order_start", 0),  # same time: reuses server 0
        ev(3.0, 0, "walkin", "fulfill_start", 1),
        ev(4.0, 1, "drive", "order_end"),
        ev(4.0, 3, "drive", "order_start", 1),
        ev(4.0, 1, "drive", "fulfill_start", 0),
        ev(5.0, 2, "walkin", "order_end"),
        ev(5.0, 2, "walkin", "fulfill_start", 0),
        ev(6.0, 3, "drive", "order_end"),
        ev(6.0, 3, "drive", "fulfill_start", 1),
        ev(8.0, 0, "walkin", "fulfill_end"),
        ev(9.0, 1, "drive", "fulfill_end"),
        ev(9.0, 2, "walkin", "fulfill_end"),
        ev(9.0, 3, "drive", "fulfill_end"),
    ]
    params = dict(DEFAULT_PARAMS, n_order_takers=2)
    log_copy = copy.deepcopy(log)

    s = state_at(log, 1.0, params)          # arrive + order_start at the same instant
    assert s["queues"]["order"] == [] and s["servers"]["order"] == [0, None]
    assert s["types"] == {0: "walkin"} and s["stats"] == {"served": 0, "in_system": 1}

    s = state_at(log, 2.7, params)          # two customers waiting, FIFO order
    assert s["queues"]["order"] == [2, 3]
    assert s["servers"]["order"] == [0, 1]
    assert s["types"] == {0: "walkin", 1: "drive", 2: "walkin", 3: "drive"}

    s = state_at(log, 3.0, params)          # three simultaneous events
    assert s["queues"]["order"] == [3]
    assert s["servers"]["order"] == [2, 1]
    assert s["servers"]["walkin_fulfill"] == [None, 0]   # server ID 1 kept from the log
    assert s["queues"]["walkin_fulfill"] == []

    s = state_at(log, 4.5, params)
    assert s["queues"]["order"] == [] and s["servers"]["order"] == [2, 3]
    assert s["servers"]["drive_fulfill"] == [1, None]    # drive customer at a drive station

    s = state_at(log, 8.0, params)
    assert s["servers"]["walkin_fulfill"] == [2, None]
    assert s["stats"] == {"served": 1, "in_system": 3} and 0 not in s["types"]

    s = state_at(log, 9.0, params)          # three departures at the same instant
    assert all(q == [] for q in s["queues"].values())
    assert all(all(c is None for c in srv) for srv in s["servers"].values())
    assert s["types"] == {} and s["stats"] == {"served": 4, "in_system": 0}
    assert log == log_copy


def test_matches_simulation_timeseries():
    for seed in (0, 1, 2):
        res = run_simulation(DEFAULT_PARAMS, seed)
        check_against_timeseries(res, DEFAULT_PARAMS, sample_times(res))


def test_other_capacities():
    for n in (2, 4):
        params = dict(DEFAULT_PARAMS, n_order_takers=n)
        res = run_simulation(params, 5)
        check_against_timeseries(res, params, sample_times(res)[::7])
        used = set()
        for t in sample_times(res)[::3]:
            used |= {i for i, c in enumerate(state_at(res["event_log"], t, params)["servers"]["order"])
                     if c is not None}
        assert used == set(range(n))


def test_simultaneous_events_heavy_load():
    # Fixed service times give many events at the same timestamp.
    params = copy.deepcopy(DEFAULT_PARAMS)
    params.update(sim_time=120, walkin_mean_iat=1.0, drive_mean_iat=1.0,
                  order_time={"low": 1, "mode": 1, "high": 1.0000001},
                  fulfill_time={"mean": 4, "sd": 0})
    res = run_simulation(params, 0)
    times = [e["t"] for e in res["event_log"]]
    tied = sorted({t for t in times if times.count(t) > 1})
    assert len(tied) > 50
    check_against_timeseries(res, params, tied)


def test_both_customer_types_and_final_state():
    res = run_simulation(DEFAULT_PARAMS, 0)
    mid = state_at(res["event_log"], 240.0, DEFAULT_PARAMS)
    assert set(mid["types"].values()) == {"walkin", "drive"}

    end = state_at(res["event_log"], res["end_time"], DEFAULT_PARAMS)
    assert all(q == [] for q in end["queues"].values())
    assert all(all(c is None for c in srv) for srv in end["servers"].values())
    assert end["types"] == {}
    assert end["stats"] == {"served": len(res["customers"]), "in_system": 0}
    # Just before the last departure, that customer is still being served.
    before = state_at(res["event_log"], res["end_time"] - 1e-9, DEFAULT_PARAMS)
    assert before["stats"]["in_system"] >= 1


def test_fake_result_compatibility():
    for seed in range(5):
        res = fake_result(seed)
        check_against_timeseries(res, DEFAULT_PARAMS, sample_times(res))


def test_repeated_calls_and_immutability():
    res = run_simulation(DEFAULT_PARAMS, 3)
    log = res["event_log"]
    log_before = copy.deepcopy(log)
    params = copy.deepcopy(DEFAULT_PARAMS)
    params_before = copy.deepcopy(params)

    forward = {t: state_at(log, t, params) for t in (10.0, 100.0, 300.0, 479.0)}
    # Backward, jumping, and repeated calls give exactly the same answers.
    for t in (479.0, 10.0, 300.0, 100.0, 10.0, 479.0):
        assert state_at(log, t, params) == forward[t]
    # Changing a returned state must not affect later calls.
    forward[100.0]["queues"]["order"].append(-1)
    assert state_at(log, 100.0, params)["queues"]["order"] != forward[100.0]["queues"]["order"]

    assert log == log_before and params == params_before
    assert DEFAULT_PARAMS == params_before


def test_bad_inputs_rejected():
    def raises(fn):
        try:
            fn()
        except (ValueError, KeyError):
            return True
        return False

    assert raises(lambda: state_at([], 0, {"n_order_takers": 3}))   # missing capacities
    out_of_range = [ev(0.5, 0, "walkin", "arrive"), ev(0.5, 0, "walkin", "order_start", 3)]
    assert raises(lambda: state_at(out_of_range, 1, DEFAULT_PARAMS))
    unsorted = [ev(2.0, 0, "walkin", "arrive"), ev(1.0, 1, "drive", "arrive")]
    assert raises(lambda: state_at(unsorted, 5, DEFAULT_PARAMS))


if __name__ == "__main__":
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    for name, fn in tests:
        fn()
        print(f"PASS  {name}")
    print(f"\nAll {len(tests)} checks passed.")
