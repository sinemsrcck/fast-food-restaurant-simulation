"""
SENG 331 - P03: Fast-Food Restaurant with Drive-Through and Walk-In
Simulation model (SimPy + NumPy).

Customer lifecycle
------------------
    Arrival -> shared order queue -> order service (order takers)
            -> type-specific fulfillment queue -> fulfillment service
            -> departure

Public API
----------
    run_simulation(params=DEFAULT_PARAMS, seed=0) -> dict
    fake_result(seed=0) -> dict

Both functions return the same result dictionary:

    {
        "timeseries": {
            "Q": {"order": [...], "walkin_fulfill": [...], "drive_fulfill": [...]},
            "B": {"order": [...], "walkin_fulfill": [...], "drive_fulfill": [...]},
            "L": [...],
        },
        "customers": [...],
        "event_log": [...],
        "end_time": float,
    }

Conventions
-----------
* All times are in minutes.
* Every timeseries is a list of (time, value) tuples. It starts with (0.0, 0)
  and gets one new tuple every time that series changes. Several tuples can
  share the same time when several events happen at the same instant. They
  are stored in the order the events happened, so the LAST tuple at a given
  time is the value right after that instant (a right-continuous step
  function).
* Server IDs are zero-based. A customer always gets the smallest free ID at
  that station.
* Arrivals are accepted only while arrival time < sim_time. Customers already
  in the system are served until they leave.
* end_time is the departure time of the last customer. If no customer
  arrives before sim_time, end_time is 0.0: the system never leaves its
  empty start state, and each timeseries holds only its starting (0.0, 0)
  point.
"""

import heapq

import numpy as np
import simpy


DEFAULT_PARAMS = {
    "sim_time": 480,
    "n_order_takers": 3,
    "n_walkin_workers": 2,
    "n_drive_workers": 2,
    "walkin_mean_iat": 2.5,
    "drive_mean_iat": 3.0,
    "order_time": {
        "low": 1,
        "mode": 2,
        "high": 3
    },
    "fulfill_time": {
        "mean": 5,
        "sd": 1.5
    }
}

CUSTOMER_TYPES = ("walkin", "drive")
STATIONS = ("order", "walkin_fulfill", "drive_fulfill")
EVENT_NAMES = ("arrive", "order_start", "order_end", "fulfill_start", "fulfill_end")


# ---------------------------------------------------------------------------
# Parameter handling
# ---------------------------------------------------------------------------

def _prepare_params(params):
    """Return a validated copy of params, with missing keys taken from
    DEFAULT_PARAMS. Neither params nor DEFAULT_PARAMS is changed."""
    p = {key: value for key, value in DEFAULT_PARAMS.items()
         if key not in ("order_time", "fulfill_time")}
    p.update({key: value for key, value in params.items()
              if key not in ("order_time", "fulfill_time")})
    # Nested dictionaries are copied separately so the caller's dicts stay untouched.
    p["order_time"] = {**DEFAULT_PARAMS["order_time"], **params.get("order_time", {})}
    p["fulfill_time"] = {**DEFAULT_PARAMS["fulfill_time"], **params.get("fulfill_time", {})}

    for key in ("n_order_takers", "n_walkin_workers", "n_drive_workers"):
        if int(p[key]) != p[key] or p[key] < 1:
            raise ValueError(f"{key} must be a positive integer, got {p[key]!r}")
        p[key] = int(p[key])
    for key in ("sim_time", "walkin_mean_iat", "drive_mean_iat"):
        if p[key] <= 0:
            raise ValueError(f"{key} must be positive, got {p[key]!r}")

    ot = p["order_time"]
    if not (0 <= ot["low"] <= ot["mode"] <= ot["high"] and ot["low"] < ot["high"]):
        raise ValueError(f"order_time needs 0 <= low <= mode <= high and low < high, got {ot}")
    ft = p["fulfill_time"]
    if ft["mean"] <= 0 or ft["sd"] < 0:
        raise ValueError(f"fulfill_time needs mean > 0 and sd >= 0, got {ft}")
    return p


def _capacities(p):
    """Number of servers at each station."""
    return {
        "order": p["n_order_takers"],
        "walkin_fulfill": p["n_walkin_workers"],
        "drive_fulfill": p["n_drive_workers"],
    }


# ---------------------------------------------------------------------------
# State recorder (used by both run_simulation and fake_result)
# ---------------------------------------------------------------------------

class _Recorder:
    """Builds the event log, the customer records and the timeseries.

    Every lifecycle event goes through log(). That one method updates the
    queue/busy/system counters, adds a timeseries point for each changed
    counter, fills in the customer's timestamp and appends to the event log.
    Because all outputs come from the same calls, they cannot disagree.
    """

    def __init__(self):
        self.q = {s: 0 for s in STATIONS}      # customers waiting at each station
        self.b = {s: 0 for s in STATIONS}      # busy servers at each station
        self.n_in_system = 0                   # L
        self.ts = {
            "Q": {s: [(0.0, 0)] for s in STATIONS},
            "B": {s: [(0.0, 0)] for s in STATIONS},
            "L": [(0.0, 0)],
        }
        self.customers = {}                    # cid -> customer record
        self.event_log = []

    def _change_q(self, t, station, delta):
        self.q[station] += delta
        self.ts["Q"][station].append((t, self.q[station]))

    def _change_b(self, t, station, delta):
        self.b[station] += delta
        self.ts["B"][station].append((t, self.b[station]))

    def _change_l(self, t, delta):
        self.n_in_system += delta
        self.ts["L"].append((t, self.n_in_system))

    def log(self, t, cid, ctype, event, server=None):
        t = float(t)
        fulfill_station = f"{ctype}_fulfill"

        if event == "arrive":
            self.customers[cid] = {
                "cid": cid, "type": ctype, "arrive": t,
                "order_start": None, "order_end": None,
                "fulfill_start": None, "fulfill_end": None,
            }
            self._change_l(t, +1)
            self._change_q(t, "order", +1)
        elif event == "order_start":
            self._change_q(t, "order", -1)
            self._change_b(t, "order", +1)
        elif event == "order_end":
            self._change_b(t, "order", -1)
            self._change_q(t, fulfill_station, +1)
        elif event == "fulfill_start":
            self._change_q(t, fulfill_station, -1)
            self._change_b(t, fulfill_station, +1)
        elif event == "fulfill_end":
            self._change_b(t, fulfill_station, -1)
            self._change_l(t, -1)
        else:
            raise ValueError(f"unknown event {event!r}")

        if event != "arrive":
            self.customers[cid][event] = t
        self.event_log.append({
            "t": t,
            "cid": cid,
            "type": ctype,
            "event": event,
            # Only start events name a server; everything else stores None.
            "server": int(server) if event in ("order_start", "fulfill_start") else None,
        })

    def result(self):
        customers = [self.customers[cid] for cid in sorted(self.customers)]
        end_time = max((c["fulfill_end"] for c in customers), default=0.0)
        return {
            "timeseries": self.ts,
            "customers": customers,
            "event_log": self.event_log,
            "end_time": float(end_time),
        }


# ---------------------------------------------------------------------------
# SimPy model
# ---------------------------------------------------------------------------

class _Station:
    """A SimPy resource plus a pool of free server IDs (smallest ID first)."""

    def __init__(self, env, capacity):
        self.resource = simpy.Resource(env, capacity=capacity)
        self.free_ids = list(range(capacity))   # a sorted list is already a valid heap

    def take_id(self):
        return heapq.heappop(self.free_ids)

    def return_id(self, server_id):
        heapq.heappush(self.free_ids, server_id)


def _sample_fulfill_time(rng, mean, sd):
    """Draw a Normal(mean, sd) duration. Negative draws are thrown away and
    drawn again."""
    while True:
        d = float(rng.normal(mean, sd))
        if d >= 0:
            return d


def _customer(env, rec, stations, cid, ctype, order_dur, fulfill_dur):
    """Takes one customer from the order queue to departure. The arrive
    event has already been logged by the arrival process."""
    # --- Order stage (one queue shared by both customer types) ---
    order = stations["order"]
    req = order.resource.request()
    yield req
    server = order.take_id()
    rec.log(env.now, cid, ctype, "order_start", server)
    yield env.timeout(order_dur)
    rec.log(env.now, cid, ctype, "order_end")
    # Return the ID before releasing, so the next waiting customer
    # (which resumes later in this same time step) can take it.
    order.return_id(server)
    order.resource.release(req)

    # --- Fulfillment stage (a separate queue for each customer type) ---
    fulfill = stations[f"{ctype}_fulfill"]
    req = fulfill.resource.request()
    yield req
    server = fulfill.take_id()
    rec.log(env.now, cid, ctype, "fulfill_start", server)
    yield env.timeout(fulfill_dur)
    rec.log(env.now, cid, ctype, "fulfill_end")
    fulfill.return_id(server)
    fulfill.resource.release(req)


def _arrivals(env, rec, stations, p, ctype, rng_iat, rng_order, rng_fulfill, next_cid):
    """Arrival process for one customer type. New customers are accepted
    only while their arrival time is strictly before sim_time."""
    mean_iat = p[f"{ctype}_mean_iat"]
    ot, ft = p["order_time"], p["fulfill_time"]
    while True:
        iat = float(rng_iat.exponential(mean_iat))
        if env.now + iat >= p["sim_time"]:
            return                               # cutoff reached: no more arrivals
        yield env.timeout(iat)

        cid = next_cid[0]
        next_cid[0] += 1
        # Both service times are drawn now, at arrival (NumPy order: left, mode, right).
        order_dur = float(rng_order.triangular(ot["low"], ot["mode"], ot["high"]))
        fulfill_dur = _sample_fulfill_time(rng_fulfill, ft["mean"], ft["sd"])

        rec.log(env.now, cid, ctype, "arrive")
        env.process(_customer(env, rec, stations, cid, ctype, order_dur, fulfill_dur))


def run_simulation(params: dict = DEFAULT_PARAMS, seed: int = 0) -> dict:
    """Run one replication of the restaurant model and return the result dict
    described in the module docstring."""
    p = _prepare_params(params)

    # Four independent NumPy streams: walk-in IAT, drive IAT, order time, fulfill time.
    rng_walkin, rng_drive, rng_order, rng_fulfill = (
        np.random.default_rng(child) for child in np.random.SeedSequence(seed).spawn(4)
    )

    env = simpy.Environment()
    rec = _Recorder()
    stations = {name: _Station(env, cap) for name, cap in _capacities(p).items()}
    next_cid = [0]   # shared customer ID counter (a list so both processes can update it)

    env.process(_arrivals(env, rec, stations, p, "walkin", rng_walkin, rng_order, rng_fulfill, next_cid))
    env.process(_arrivals(env, rec, stations, p, "drive", rng_drive, rng_order, rng_fulfill, next_cid))

    # Run with no time limit. The arrival processes stop at sim_time, so the
    # run ends when the last customer has left.
    env.run()

    return rec.result()


# ---------------------------------------------------------------------------
# Synthetic result for the other team members
# ---------------------------------------------------------------------------

# When several events share a timestamp, sort them by this rank. Departures
# and stage ends come first so their servers are free before new starts.
# Each customer's own events still come out in lifecycle order:
# arrive(2) < order_start(3), order_end(1) < fulfill_start(4), and an
# order_start never ties with its own order_end because order time >= low > 0.
_TIE_RANK = {"fulfill_end": 0, "order_end": 1, "arrive": 2, "order_start": 3, "fulfill_start": 4}


def _fcfs_schedule(ready_times, durations, n_servers):
    """Assign customers (already in FCFS order) to a multi-server station.

    Returns (start_times, server_ids). A customer takes the smallest free ID
    when it arrives. If no server is free, it waits for the first one to
    finish (lowest ID on ties).
    """
    free_at = [0.0] * n_servers
    starts, servers = [], []
    for ready, dur in zip(ready_times, durations):
        idle = [s for s in range(n_servers) if free_at[s] <= ready]
        s = idle[0] if idle else min(range(n_servers), key=lambda k: (free_at[k], k))
        start = max(ready, free_at[s])
        free_at[s] = start + dur
        starts.append(start)
        servers.append(s)
    return starts, servers


def fake_result(seed: int = 0) -> dict:
    """Return a small, reproducible, self-consistent synthetic result.

    This does not run SimPy. It builds an FCFS schedule directly with the
    default station capacities, then sends the events through the same
    _Recorder that run_simulation uses, so the schema, timeseries and event
    log match the real model's.
    """
    rng = np.random.default_rng(seed)
    cap = _capacities(DEFAULT_PARAMS)
    ot, ft = DEFAULT_PARAMS["order_time"], DEFAULT_PARAMS["fulfill_time"]

    # 20 customers, both types. The first 6 arrive within the first minute, so
    # there are more customers than the 3 order takers and the order queue
    # waits are guaranteed to be nonzero. The rest arrive over the next 30 minutes.
    n_burst, n_rest = 6, 14
    arrivals = np.concatenate([
        np.sort(rng.uniform(0.0, 1.0, n_burst)),
        np.sort(rng.uniform(1.0, 31.0, n_rest)),
    ])
    types = ["walkin" if i % 2 == 0 else "drive" for i in range(n_burst)]  # both types in the burst
    types += [str(t) for t in rng.choice(CUSTOMER_TYPES, size=n_rest)]
    order_durs = rng.triangular(ot["low"], ot["mode"], ot["high"], size=len(arrivals))
    fulfill_durs = np.abs(rng.normal(ft["mean"], ft["sd"], size=len(arrivals)))

    customers = [
        {"cid": i, "type": types[i], "arrive": float(arrivals[i]),
         "order_dur": float(order_durs[i]), "fulfill_dur": float(fulfill_durs[i])}
        for i in range(len(arrivals))
    ]

    # Order stage: one shared queue, FCFS by arrival.
    starts, servers = _fcfs_schedule(
        [c["arrive"] for c in customers], [c["order_dur"] for c in customers], cap["order"])
    for c, start, s in zip(customers, starts, servers):
        c["order_start"], c["order_server"] = start, s
        c["order_end"] = start + c["order_dur"]

    # Fulfillment stage: one FCFS queue per type, in order_end order.
    for ctype in CUSTOMER_TYPES:
        group = sorted((c for c in customers if c["type"] == ctype),
                       key=lambda c: (c["order_end"], c["cid"]))
        starts, servers = _fcfs_schedule(
            [c["order_end"] for c in group], [c["fulfill_dur"] for c in group],
            cap[f"{ctype}_fulfill"])
        for c, start, s in zip(group, starts, servers):
            c["fulfill_start"], c["fulfill_server"] = start, s
            c["fulfill_end"] = start + c["fulfill_dur"]

    # Build every event, sort by time and tie rank, then send them through the recorder.
    events = []
    for c in customers:
        events += [
            (c["arrive"], "arrive", c, None),
            (c["order_start"], "order_start", c, c["order_server"]),
            (c["order_end"], "order_end", c, None),
            (c["fulfill_start"], "fulfill_start", c, c["fulfill_server"]),
            (c["fulfill_end"], "fulfill_end", c, None),
        ]
    events.sort(key=lambda e: (e[0], _TIE_RANK[e[1]], e[2]["cid"]))

    rec = _Recorder()
    for t, event, c, server in events:
        rec.log(t, c["cid"], c["type"], event, server)
    return rec.result()
