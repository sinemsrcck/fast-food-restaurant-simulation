"""
SENG 331 - P03: Fast-Food Restaurant with Drive-Through and Walk-In
Replay engine: rebuilds the restaurant state at any time from the event log.

    state_at(event_log, t, params) -> dict

The event log is the one produced by model.run_simulation() / model.fake_result():
a chronologically ordered list of

    {"t": float, "cid": int, "type": "walkin" | "drive",
     "event": "arrive" | "order_start" | "order_end" | "fulfill_start" | "fulfill_end",
     "server": int | None}

state_at() rebuilds the state from scratch on every call by replaying the
events with time <= t. It never remembers anything between calls, so it can
be called at any time in any order (forward, backward, jumping). It has no
Pygame or other visualisation code.
"""

STATIONS = ("order", "walkin_fulfill", "drive_fulfill")
CUSTOMER_TYPES = ("walkin", "drive")

# Parameter key that holds each station's number of servers.
_CAPACITY_KEYS = {
    "order": "n_order_takers",
    "walkin_fulfill": "n_walkin_workers",
    "drive_fulfill": "n_drive_workers",
}


def _capacities(params):
    """Number of servers at each station, read from params (never modified)."""
    caps = {}
    for station, key in _CAPACITY_KEYS.items():
        if key not in params:
            raise KeyError(f"params is missing {key!r}, needed for the {station} station size")
        n = params[key]
        if int(n) != n or n < 1:
            raise ValueError(f"{key} must be a positive integer, got {n!r}")
        caps[station] = int(n)
    return caps


def state_at(event_log, t, params) -> dict:
    """Return the restaurant state right after all events with time <= t.

    Returned dictionary:
        "t"       : the requested time, unchanged
        "queues"  : station -> list of waiting customer IDs, front of the queue first
        "servers" : station -> fixed-length list, index = zero-based server ID,
                    value = customer ID being served there, or None if idle
        "types"   : customer ID -> "walkin" / "drive", for every customer
                    currently in the system (in a queue or being served)
        "stats"   : {"served": customers who have departed by time t,
                     "in_system": customers currently in the system}

    Events that share a timestamp are applied in their order in the log, which
    is the causal order. A time before the first event (or an empty log) gives
    the empty start state. A time at or after the last event gives the final
    state.
    """
    caps = _capacities(params)
    queues = {s: [] for s in STATIONS}
    servers = {s: [None] * caps[s] for s in STATIONS}
    types = {}                 # customers currently in the system
    where_served = {}          # cid -> (station, server id) while being served
    served = 0

    prev_t = float("-inf")
    for e in event_log:
        et = e["t"]
        if et < prev_t:
            raise ValueError(f"event log is not in chronological order at {e}")
        prev_t = et
        if et > t:
            break              # the log is sorted, so every later event is also > t

        cid, ctype, event = e["cid"], e["type"], e["event"]
        if ctype not in CUSTOMER_TYPES:
            raise ValueError(f"unknown customer type in {e}")
        fulfill = f"{ctype}_fulfill"

        if event == "arrive":
            types[cid] = ctype
            queues["order"].append(cid)
        elif event == "order_start":
            _leave_queue(queues["order"], cid, e)
            _occupy(servers, where_served, "order", e)
        elif event == "order_end":
            _free(servers, where_served, "order", e)
            queues[fulfill].append(cid)
        elif event == "fulfill_start":
            _leave_queue(queues[fulfill], cid, e)
            _occupy(servers, where_served, fulfill, e)
        elif event == "fulfill_end":
            _free(servers, where_served, fulfill, e)
            del types[cid]
            served += 1
        else:
            raise ValueError(f"unknown event name in {e}")

    return {
        "t": t,
        "queues": queues,
        "servers": servers,
        "types": types,
        "stats": {"served": served, "in_system": len(types)},
    }


# ---------------------------------------------------------------------------
# Helpers. Each one checks that the log is consistent, so a broken log fails
# loudly instead of drawing a wrong picture.
# ---------------------------------------------------------------------------

def _leave_queue(queue, cid, e):
    """Remove a customer from its queue. With FCFS this is always the front;
    removing by ID keeps it correct even if it were not."""
    if cid not in queue:
        raise ValueError(f"customer {cid} starts service without waiting in the queue: {e}")
    queue.remove(cid)


def _occupy(servers, where_served, station, e):
    sid = e["server"]
    if not isinstance(sid, int) or not 0 <= sid < len(servers[station]):
        raise ValueError(f"server ID {sid!r} is outside 0..{len(servers[station]) - 1} "
                         f"for the {station} station: {e}")
    if servers[station][sid] is not None:
        raise ValueError(f"{station} server {sid} is already busy with customer "
                         f"{servers[station][sid]}: {e}")
    servers[station][sid] = e["cid"]
    where_served[e["cid"]] = (station, sid)


def _free(servers, where_served, station, e):
    # End events have server=None in the log, so look up which server this customer was using.
    loc = where_served.pop(e["cid"], None)
    if loc is None or loc[0] != station:
        raise ValueError(f"customer {e['cid']} ends service at {station} without starting it: {e}")
    servers[station][loc[1]] = None
