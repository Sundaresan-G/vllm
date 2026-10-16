#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Merge vLLM per-subsystem step timings into one time-sorted view.

Option-1 correlation: every subsystem (APIServer, EngineCore, GPU Worker,
GPU model runner) logs inner UTC timestamps on the same node clock. We parse
those inner timestamps (NOT the shared logger-flush prefix), sort all records
on one timeline, and also join them onto EngineCore's ``batch=`` counter --
the most authoritative per-microbatch id -- by nearest dispatch timestamp.

Usage:
    /usr/bin/python merge_timings.py <logfile|logdir> [--lo STEP] [--hi STEP]
    /usr/bin/python merge_timings.py <logfile|logdir> --timeline \
        --from HH:MM:SS --to HH:MM:SS

<logfile> may be a single combined log (legacy, all ranks on stdout) or a
directory of per-rank files produced by ``VLLM_STEP_TIMING_LOG_DIR`` (e.g.
``step_timings/step_timing_Worker_PP0_TP3.log``). Reading per-rank files is
preferred: each file holds one rank's records with no cross-rank line
interleaving, so no dumps are corrupted.
"""

import argparse
import os
from bisect import bisect_left
from datetime import datetime, timedelta

import regex as re

TS = r"(\d{2}-\d{2} \d{2}:\d{2}:\d{2}\.\d+)"


def ts(s: str) -> datetime:
    return datetime.strptime("2026-" + s, "%Y-%m-%d %H:%M:%S.%f")


def hms(t: datetime) -> str:
    return t.strftime("%H:%M:%S.%f")


# ---- per-subsystem record parsers -------------------------------------------
# All four subsystems now emit a uniform multi-line format: a header line
# followed by indented phase lines, each phase carrying its own cpu_start /
# cpu_end wall timestamps (and gpu_duration_ms where GPU events exist).
RE_EC_CALL = re.compile(r"engine invocation timing call=(\d+) mode=(\w+)")
RE_EC_DISP_HDR = re.compile(
    r"dispatch batch=(\d+) num_tokens=(\d+) "
    r"cpu_start=" + TS + r" cpu_end=" + TS + r" cpu_duration_ms=([\d.]+)"
)
RE_EC_COLL_HDR = re.compile(
    r"collect\[future.result\(\)\] batch=(\d+) "
    r"cpu_start=" + TS + r" cpu_end=" + TS + r" cpu_duration_ms=([\d.]+) "
    r"queued_before_collect=([\d.]+) batch_end_to_end=([\d.]+)"
)
RE_WORKER_HDR = re.compile(
    r"Worker_(PP\d+)_TP(\d+)[^)]*\).*?Worker execute_model step timing "
    r"step=(\d+) num_tokens=(\d+)"
)
RE_GPU_HDR = re.compile(
    r"Worker_(PP\d+)_TP(\d+)[^)]*\).*?GPU model runner step timing "
    r"step=(\d+) num_tokens=(\d+) decode=(\w+)"
)
RE_API_HDR = re.compile(
    r"API server step timing step=(\d+) engine=(\d+) "
    r"num_request_outputs=(\d+) num_output_token_ids=(\d+) "
    r"cpu_start=" + TS + r" cpu_end=" + TS + r" cpu_duration_ms=([\d.]+)"
)
# Generic indented phase line, shared by every subsystem.
RE_PHASE = re.compile(
    r"^\s+(\w+) cpu_start="
    + TS
    + r" cpu_end="
    + TS
    + r" cpu_duration_ms=([\d.]+)(?: gpu_duration_ms=([\d.]+))?"
)

# Canonical worker phase order for the compact phases string.
_WORKER_PHASE_ORDER = (
    "prev_send_wait",
    "irecv_post",
    "execute",
    "pp_sync_recv",
    "pp_recv_wait",
    "pp_tp_all_gather",
    "sp_residual_all_gather",
    "isend_post",
)


def _content(line: str) -> str:
    """Strip the logger prefix (`... ] `) leaving the inner message, keeping
    any leading indentation that distinguishes phase lines from headers."""
    i = line.find("] ")
    return line[i + 2 :] if i != -1 else line


def _phase_ms(phases: dict, name: str) -> str:
    ph = phases.get(name)
    return f"{ph['cpu_ms']:.3f}" if ph else "-"


def _phase_gpu(phases: dict, name: str) -> str:
    ph = phases.get(name)
    if ph and ph.get("gpu_ms") is not None:
        return f"{ph['gpu_ms']:.3f}"
    return "-"


def _worker_phase_str(phases: dict) -> str:
    parts = []
    for name in _WORKER_PHASE_ORDER:
        ph = phases.get(name)
        if not ph:
            continue
        field = f"{name}={ph['cpu_ms']:.3f}"
        if ph.get("gpu_ms") is not None:
            field += f"(gpu{ph['gpu_ms']:.3f})"
        parts.append(field)
    return " ".join(parts)


def _span(phases: dict):
    """Return (min phase start, max phase end) across a record's phases."""
    starts = [p["start"] for p in phases.values()]
    ends = [p["end"] for p in phases.values()]
    return (min(starts) if starts else None, max(ends) if ends else None)


def _iter_log_lines(path: str):
    """Yield lines from a single log file or every ``*.log`` in a directory.

    A directory of per-rank files (``VLLM_STEP_TIMING_LOG_DIR`` output) is read
    file-by-file so each rank's records stay contiguous and uncorrupted.
    """
    if os.path.isdir(path):
        files = sorted(
            os.path.join(path, f) for f in os.listdir(path) if f.endswith(".log")
        )
        for fp in files:
            with open(fp) as fh:
                yield from fh
    else:
        with open(path) as fh:
            yield from fh


def parse(path: str):
    """Parse one run log into raw records.

    Worker/GPU records are keyed by ``(pp, tp, step)`` so every tensor-parallel
    rank is preserved; collapse to one record per ``(pp, step)`` happens later
    in :func:`reduce_tp`. Returns raw dicts (no timeline); call
    :func:`build_events` after reducing TP to get the unified event list.
    """
    ec_dispatch = {}  # batch -> {...}
    ec_collect = {}  # batch -> {...}
    worker = {}  # (pp, tp, step) -> {...}
    gpu = {}  # (pp, tp, step) -> {...}
    api = {}  # step -> {...}

    cur = None  # record dict currently receiving phase lines
    for line in _iter_log_lines(path):
        if RE_EC_CALL.search(line):
            cur = None  # new invocation group; wait for dispatch/collect header
            continue
        m = RE_EC_DISP_HDR.search(line)
        if m:
            b = int(m.group(1))
            rec = {
                "batch": b,
                "num_tokens": int(m.group(2)),
                "start": ts(m.group(3)),
                "end": ts(m.group(4)),
                "cpu_ms": float(m.group(5)),
                "phases": {},
            }
            ec_dispatch[b] = rec
            cur = rec
            continue
        m = RE_EC_COLL_HDR.search(line)
        if m:
            b = int(m.group(1))
            rec = {
                "batch": b,
                "start": ts(m.group(2)),
                "end": ts(m.group(3)),
                "cpu_ms": float(m.group(4)),
                "queued_ms": float(m.group(5)),
                "e2e_ms": float(m.group(6)),
                "phases": {},
            }
            ec_collect[b] = rec
            cur = rec
            continue
        m = RE_WORKER_HDR.search(line)
        if m:
            pp, tp, step = m.group(1), int(m.group(2)), int(m.group(3))
            rec = {
                "pp": pp,
                "tp": tp,
                "rank": f"{pp}_TP{tp}",
                "step": step,
                "num_tokens": int(m.group(4)),
                "phases": {},
            }
            worker[(pp, tp, step)] = rec
            cur = rec
            continue
        m = RE_GPU_HDR.search(line)
        if m:
            pp, tp, step = m.group(1), int(m.group(2)), int(m.group(3))
            rec = {
                "pp": pp,
                "tp": tp,
                "rank": f"{pp}_TP{tp}",
                "step": step,
                "num_tokens": int(m.group(4)),
                "decode": m.group(5),
                "phases": {},
            }
            gpu[(pp, tp, step)] = rec
            cur = rec
            continue
        m = RE_API_HDR.search(line)
        if m:
            step = int(m.group(1))
            rec = {
                "step": step,
                "engine": int(m.group(2)),
                "num_request_outputs": int(m.group(3)),
                "num_output_token_ids": int(m.group(4)),
                "start": ts(m.group(5)),
                "end": ts(m.group(6)),
                "cpu_ms": float(m.group(7)),
                "phases": {},
            }
            api[step] = rec
            cur = rec
            continue
        m = RE_PHASE.search(_content(line))
        if m and cur is not None:
            cur["phases"][m.group(1)] = {
                "start": ts(m.group(2)),
                "end": ts(m.group(3)),
                "cpu_ms": float(m.group(4)),
                "gpu_ms": float(m.group(5)) if m.group(5) else None,
            }
            continue

    # Finalize derived fields on every per-TP record (timeline is built
    # later, in build_events, once TP ranks have been reduced).
    for rec in worker.values():
        _finalize_worker_rec(rec)
    for rec in gpu.values():
        _finalize_gpu_rec(rec)
    return ec_dispatch, ec_collect, worker, gpu, api


def _finalize_worker_rec(rec):
    start, end = _span(rec["phases"])
    rec["start"], rec["end"] = start, end
    rec["phase_str"] = _worker_phase_str(rec["phases"])


def _finalize_gpu_rec(rec):
    phs = rec["phases"]
    pre = phs.get("preprocess")
    rec["pre_start"] = pre["start"] if pre else None
    rec["pre_gpu"] = _phase_gpu(phs, "preprocess")
    rec["fwd_gpu"] = _phase_gpu(phs, "forward")
    rec["post_gpu"] = _phase_gpu(phs, "postprocess")
    rec["samp_gpu"] = _phase_gpu(phs, "sample")
    rec["upd_gpu"] = _phase_gpu(phs, "update_states")
    rec["ppbc_gpu"] = _phase_gpu(phs, "pp_broadcast")
    rec["pprecv_gpu"] = _phase_gpu(phs, "pp_recv")
    rec["upd_cpu"] = _phase_ms(phs, "update_states")
    rec["ppbc_cpu"] = _phase_ms(phs, "pp_broadcast")
    rec["pprecv_cpu"] = _phase_ms(phs, "pp_recv")
    rec["kvpf_cpu"] = _phase_ms(phs, "kv_post_forward")
    rec["kvpf_gpu"] = _phase_gpu(phs, "kv_post_forward")
    start, end = _span(phs)
    rec["start"], rec["end"] = start, end
    # Split into the execute_model portion (pre/fwd/post, logged by the
    # worker) and the later sample_tokens portion (sample/pp_broadcast/
    # update_states or pp_recv), which runs after execute_model returns.
    exec_phs = {
        k: v for k, v in phs.items() if k in ("preprocess", "forward", "postprocess")
    }
    samp_phs = {
        k: v
        for k, v in phs.items()
        if k
        in (
            "sample",
            "pp_broadcast",
            "update_states",
            "pp_recv",
            "kv_post_forward",
        )
    }
    rec["exec_span"] = _span(exec_phs)
    rec["samp_span"] = _span(samp_phs)


def reduce_tp(records, mode, tp_sel=0):
    """Collapse per-TP records ``(pp, tp, step)`` to one per ``(pp, step)``.

    Modes:
      * ``select``   keep the record for TP rank ``tp_sel`` (default 0).
      * ``straggler``/``max``  slowest-finishing rank (latest record end);
        best reflects collective phases (pp_broadcast / all_gather), which
        cannot complete until the last rank arrives.
      * ``leader``/``min``   fastest-finishing rank (earliest record end).
      * ``last``   last rank logged for that step (legacy behavior).
    """
    groups = {}
    for (pp, tp, step), rec in records.items():
        groups.setdefault((pp, step), []).append((tp, rec))
    out = {}
    for (pp, step), lst in groups.items():
        if mode == "select":
            chosen = next((r for tp, r in lst if tp == tp_sel), None)
            if chosen is None:
                continue
        elif mode in ("straggler", "max"):
            chosen = max(lst, key=lambda tr: tr[1].get("end") or datetime.min)[1]
        elif mode in ("leader", "min"):
            chosen = min(lst, key=lambda tr: tr[1].get("end") or datetime.max)[1]
        else:  # last
            chosen = lst[-1][1]
        out[(pp, step)] = chosen
    return out


def build_events(ec_dispatch, ec_collect, worker, gpu, api):
    """Build the time-sorted event list from reduced (per-PP) records."""
    events = []  # (time, tag, dict) on one timeline
    for rec in ec_dispatch.values():
        events.append((rec["start"], "EC.dispatch", rec))
    for rec in ec_collect.values():
        events.append((rec["start"], "EC.collect_beg", rec))
        events.append((rec["end"], "EC.collect_end", rec))
    for (pp, _step), rec in worker.items():
        if rec["start"] is not None:
            events.append((rec["start"], f"W.{pp}.beg", rec))
            events.append((rec["end"], f"W.{pp}.end", rec))
    for (pp, _step), rec in gpu.items():
        e_start, e_end = rec["exec_span"]
        if e_start is not None:
            events.append((e_start, f"G.{pp}.exec.beg", rec))
            events.append((e_end, f"G.{pp}.exec.end", rec))
        s_start, s_end = rec["samp_span"]
        if s_start is not None:
            events.append((s_start, f"G.{pp}.samp.beg", rec))
            events.append((s_end, f"G.{pp}.samp.end", rec))
    for rec in api.values():
        events.append((rec["start"], "API.beg", rec))
        events.append((rec["end"], "API.end", rec))
    events.sort(key=lambda e: e[0])
    return events


def nearest(sorted_times, keys, target, tol_ms=50.0):
    """Return key whose time is closest to target within tol."""
    if not sorted_times:
        return None
    i = bisect_left(sorted_times, target)
    best, bestd = None, None
    for j in (i - 1, i):
        if 0 <= j < len(sorted_times):
            d = abs((sorted_times[j] - target).total_seconds()) * 1000
            if bestd is None or d < bestd:
                bestd, best = d, keys[j]
    if bestd is not None and bestd <= tol_ms:
        return best
    return None


def join_on_batch(ec_dispatch, ec_collect, worker, gpu, api, lo, hi):
    # Build sorted lookup lists for nearest-time matching.
    def sorted_by(d, tkey):
        items = sorted(
            (r for r in d.values() if r.get(tkey) is not None),
            key=lambda r: r[tkey],
        )
        return [r[tkey] for r in items], items

    pp0w = {k: v for k, v in worker.items() if k[0] == "PP0"}
    pp1w = {k: v for k, v in worker.items() if k[0] == "PP1"}
    pp0w_t, pp0w = sorted_by(pp0w, "start")
    pp1w_t, pp1w = sorted_by(pp1w, "start")
    g0_t, g0 = sorted_by({k: v for k, v in gpu.items() if k[0] == "PP0"}, "pre_start")
    g1_t, g1 = sorted_by({k: v for k, v in gpu.items() if k[0] == "PP1"}, "pre_start")
    api_t, api_l = sorted_by(api, "start")

    print(
        f"{'ECbatch':>7} {'ntok':>4} {'dispatch':>15} | "
        f"{'PP0.step':>8} {'PP0.fwdgpu':>10} | {'PP1.step':>8} "
        f"{'PP1.fwdgpu':>10} {'PP1.sampgpu':>11} "
        f"{'PP1.ppbc_cpu':>12} {'PP1.ppbc_gpu':>12} | "
        f"{'EC.wait_ms':>10} | {'API.step':>8} {'API.out':>7}"
    )
    for b in sorted(ec_dispatch):
        d = ec_dispatch[b]
        if not (lo <= b <= hi):
            continue
        t = d["start"]
        i0 = nearest(pp0w_t, pp0w, t)
        ig0 = nearest(g0_t, g0, t)
        # PP1 and API happen slightly later; widen tolerance.
        i1 = nearest(pp1w_t, pp1w, t, tol_ms=80.0)
        ig1 = nearest(g1_t, g1, t, tol_ms=80.0)
        c = ec_collect.get(b, {})
        ia = nearest(api_t, api_l, t, tol_ms=120.0)
        wait_ms = _phase_ms(c.get("phases", {}), "model_output_future_wait")

        def g(x, k):
            return str(x.get(k, "-")) if x else "-"

        print(
            f"{b:>7} {d['num_tokens']:>4} {hms(t):>15} | "
            f"{(i0['step'] if i0 else '-'):>8} {g(ig0, 'fwd_gpu'):>10} | "
            f"{(i1['step'] if i1 else '-'):>8} {g(ig1, 'fwd_gpu'):>10} "
            f"{g(ig1, 'samp_gpu'):>11} "
            f"{g(ig1, 'ppbc_cpu'):>12} {g(ig1, 'ppbc_gpu'):>12} | "
            f"{wait_ms:>10} | "
            f"{(ia['step'] if ia else '-'):>8} "
            f"{(ia['num_request_outputs'] if ia else '-'):>7}"
        )


def timeline(events, t_from, t_to):
    """t_from/t_to may be HH:MM:SS strings or datetimes (or None)."""
    for t, tag, rec in events:
        hh = hms(t)
        lo = hms(t_from) if isinstance(t_from, datetime) else t_from
        hi = hms(t_to) if isinstance(t_to, datetime) else t_to
        if lo and hh < lo:
            continue
        if hi and hh > hi:
            continue
        phs = rec.get("phases", {})
        if tag == "EC.dispatch":
            extra = (
                f"batch={rec['batch']} ntok={rec['num_tokens']} "
                f"sched={_phase_ms(phs, 'schedule_cpu')} "
                f"enqueue={_phase_ms(phs, 'execute_model_rpc_enqueue')}"
            )
        elif tag == "EC.collect_beg":
            extra = f"batch={rec['batch']} (blocks)"
        elif tag == "EC.collect_end":
            extra = (
                f"batch={rec['batch']} "
                f"wait_ms={_phase_ms(phs, 'model_output_future_wait')} "
                f"update={_phase_ms(phs, 'scheduler_update_from_output_cpu')} "
                f"(returns)"
            )
        elif tag.endswith(".beg") and tag.startswith("W."):
            extra = f"step={rec['step']} ntok={rec['num_tokens']}"
        elif tag.endswith(".end") and tag.startswith("W."):
            extra = f"step={rec['step']} ntok={rec['num_tokens']} (returns)"
            if rec.get("phase_str"):
                extra += f" | {rec['phase_str']}"
        elif tag.startswith("G.") and ".exec." in tag:
            if tag.endswith(".beg"):
                extra = f"step={rec['step']} ntok={rec['num_tokens']}"
            else:
                extra = (
                    f"step={rec['step']} ntok={rec['num_tokens']} (exec done) "
                    f"pre_gpu={rec.get('pre_gpu', '-')} "
                    f"fwd_gpu={rec.get('fwd_gpu', '-')} "
                    f"post_gpu={rec.get('post_gpu', '-')}"
                )
        elif tag.startswith("G.") and ".samp." in tag:
            if tag.endswith(".beg"):
                extra = f"step={rec['step']} ntok={rec['num_tokens']}"
            else:
                extra = (
                    f"step={rec['step']} ntok={rec['num_tokens']} (samp done) "
                    f"samp_gpu={rec.get('samp_gpu', '-')} "
                    f"upd_cpu={rec.get('upd_cpu', '-')} "
                    f"upd_gpu={rec.get('upd_gpu', '-')} "
                    f"ppbc_cpu={rec.get('ppbc_cpu', '-')} "
                    f"ppbc_gpu={rec.get('ppbc_gpu', '-')} "
                    f"pprecv_cpu={rec.get('pprecv_cpu', '-')} "
                    f"pprecv_gpu={rec.get('pprecv_gpu', '-')} "
                    f"kvpf_cpu={rec.get('kvpf_cpu', '-')} "
                    f"kvpf_gpu={rec.get('kvpf_gpu', '-')}"
                )
        elif tag == "API.beg":
            extra = (
                f"step={rec['step']} out={rec['num_request_outputs']} "
                f"tok={rec['num_output_token_ids']} (blocks)"
            )
        elif tag == "API.end":
            extra = (
                f"step={rec['step']} out={rec['num_request_outputs']} "
                f"tok={rec['num_output_token_ids']} (returns) "
                f"recv={_phase_ms(phs, 'receive_wait')}"
            )
        else:
            extra = ""
        print(f"{hh}  {tag:<15} {extra}")


def _exec_gpu_ms(rec: dict):
    """Total GPU time of the execute_model phases (pre/fwd/post), or None."""
    phs = rec.get("phases", {})
    total = 0.0
    found = False
    for name in ("preprocess", "forward", "postprocess"):
        ph = phs.get(name)
        if ph and ph.get("gpu_ms") is not None:
            total += ph["gpu_ms"]
            found = True
    return total if found else None


def _intervals(events, splits=False):
    """Pair .beg/.end (and _beg/_end) events into intervals.

    Returns a list of (lane, start, end, rec). Point events (EC.dispatch)
    are skipped. The lane is the tag with its trailing ``.beg``/``_end``
    suffix removed, e.g. ``G.PP0.exec`` or ``EC.collect``. For each exec
    interval an extra ``.gpu`` lane is emitted that shares the CPU start
    but spans the actual GPU duration (pre/fwd/post), since forward runs
    asynchronously after the CPU-side execute returns.

    When ``splits`` is set, break the worker and sample records into their
    comms sub-phases (irecv/isend/prev_send_wait and pp_broadcast/pp_recv)
    as their own lanes, using each phase's own cpu_start/cpu_end.
    """
    pending = {}
    intervals = []
    for t, tag, rec in events:
        if tag == "EC.dispatch":
            continue
        lane = tag[:-4]
        if tag.endswith(".beg") or tag.endswith("_beg"):
            pending[(lane, id(rec))] = t
        elif tag.endswith(".end") or tag.endswith("_end"):
            start = pending.pop((lane, id(rec)), None)
            if start is not None:
                intervals.append((lane, start, t, rec))
                if lane.endswith(".exec"):
                    gms = _exec_gpu_ms(rec)
                    if gms is not None:
                        gend = start + timedelta(milliseconds=gms)
                        intervals.append((lane + ".gpu", start, gend, rec))
                if splits:
                    intervals += _phase_split_intervals(lane, rec)
    return intervals


# Phase name -> lane suffix for the comms sub-lanes (--splits). Worker
# phases hang off the ``W.PPx`` lane; sample phases off the ``G.PPx`` base.
_WORKER_PHASE_LANES = (
    ("prev_send_wait", "psw"),
    ("irecv_post", "irecv"),
    ("isend_post", "isend"),
)
_GPU_PHASE_LANES = (
    ("pp_recv", "pprecv"),
    ("pp_broadcast", "ppbc"),
)


def _phase_split_intervals(lane: str, rec: dict):
    """Emit per-phase intervals for a worker/sample record (--splits)."""
    phs = rec.get("phases", {})
    out = []
    if lane.startswith("W."):
        for pname, suf in _WORKER_PHASE_LANES:
            ph = phs.get(pname)
            if ph:
                out.append((f"{lane}.{suf}", ph["start"], ph["end"], rec))
    elif lane.endswith(".samp"):
        base = lane[:-5]
        for pname, suf in _GPU_PHASE_LANES:
            ph = phs.get(pname)
            if ph:
                out.append((f"{base}.{suf}", ph["start"], ph["end"], rec))
    return out


# Lane rows, top to bottom. Lanes not listed are appended sorted.
_GANTT_LANE_ORDER = (
    "API",
    "EC.collect",
    "W.PP0",
    "W.PP0.psw",
    "W.PP0.irecv",
    "W.PP0.isend",
    "G.PP0.exec",
    "G.PP0.exec.gpu",
    "G.PP0.samp",
    "G.PP0.pprecv",
    "G.PP0.ppbc",
    "W.PP1",
    "W.PP1.psw",
    "W.PP1.irecv",
    "W.PP1.isend",
    "G.PP1.exec",
    "G.PP1.exec.gpu",
    "G.PP1.samp",
    "G.PP1.pprecv",
    "G.PP1.ppbc",
)


def _gantt_char(lane: str) -> str:
    if lane.endswith(".irecv"):
        return "<"
    if lane.endswith(".isend"):
        return ">"
    if lane.endswith(".psw"):
        return "."
    if lane.endswith(".ppbc"):
        return "B"
    if lane.endswith(".pprecv"):
        return "R"
    if lane.endswith(".exec.gpu"):
        return "#"
    if lane.endswith(".exec"):
        return "="
    if lane.endswith(".samp"):
        return "*"
    if lane.startswith("W."):
        return "-"
    if lane == "API":
        return "~"
    if lane == "EC.collect":
        return "+"
    return "#"


def _interval_label(lane: str, rec: dict) -> str:
    if lane == "EC.collect":
        return f"b{rec.get('batch', '?')}"
    if "step" in rec:
        return f"s{rec['step']}"
    return ""


def gantt(events, t_from, t_to, width=120, splits=False):
    """Render an ASCII Gantt chart of the begin/end intervals.

    t_from/t_to may be datetimes, HH:MM:SS strings, or None (auto from the
    intervals present). Each lane is one row; time maps linearly to columns.
    """
    lo = hms(t_from) if isinstance(t_from, datetime) else t_from
    hi = hms(t_to) if isinstance(t_to, datetime) else t_to
    ivs = []
    for lane, s, e, rec in _intervals(events, splits=splits):
        if lo and hms(e) < lo:
            continue
        if hi and hms(s) > hi:
            continue
        # Drop ntok=0 bubble steps; they add near-zero-width bars that only
        # collide with the real decode bars and their labels.
        if rec.get("num_tokens") == 0:
            continue
        ivs.append((lane, s, e, rec))
    if not ivs:
        print("# gantt: no intervals in window")
        return

    t0 = t_from if isinstance(t_from, datetime) else min(s for _, s, _, _ in ivs)
    t1 = t_to if isinstance(t_to, datetime) else max(e for _, _, e, _ in ivs)
    span = (t1 - t0).total_seconds() * 1000.0
    if span <= 0:
        span = 1.0

    def col(t):
        c = int((t - t0).total_seconds() * 1000.0 / span * (width - 1))
        return max(0, min(width - 1, c))

    lanes = {}
    for lane, s, e, rec in ivs:
        lanes.setdefault(lane, []).append((s, e, rec))
    order = [ln for ln in _GANTT_LANE_ORDER if ln in lanes]
    order += sorted(ln for ln in lanes if ln not in _GANTT_LANE_ORDER)
    label_w = max(len(ln) for ln in order)

    print(
        f"# gantt {hms(t0)} .. {hms(t1)}  span={span:.3f} ms  "
        f"{width} cols ({span / width:.4f} ms/col)"
    )
    for lane in order:
        row = [" "] * width
        ch = _gantt_char(lane)
        for s, e, rec in sorted(lanes[lane], key=lambda x: x[0]):
            cs, ce = col(s), col(e)
            if ce < cs:
                ce = cs
            for c in range(cs, ce + 1):
                row[c] = ch
            lbl = _interval_label(lane, rec)
            # Only stamp the label when it fits entirely within the bar,
            # so near-zero-width phases don't leave a truncated stub (e.g.
            # a lone "s" from "s12" on a 1-column prev_send_wait bar).
            if lbl and cs + len(lbl) - 1 <= ce and cs + len(lbl) <= width:
                for i, lc in enumerate(lbl):
                    row[cs + i] = lc
        print(f"{lane:<{label_w}} |{''.join(row)}|")

    # Bottom axis with ms tick labels.
    marks = min(8, width // 12) or 1
    tick = [" "] * width
    positions = []
    for k in range(marks + 1):
        c = int(k / marks * (width - 1))
        tick[c] = "|"
        positions.append((c, f"{span * k / marks:.1f}"))
    print(f"{'':<{label_w}} |{''.join(tick)}|")
    lbl_row = [" "] * (width + 16)
    for c, txt in positions:
        for i, lc in enumerate(txt):
            if c + i < len(lbl_row):
                lbl_row[c + i] = lc
    print(f"{'':<{label_w}}  {''.join(lbl_row).rstrip()}  (ms)")


def _dur_ms(a, b):
    return (b - a).total_seconds() * 1000.0


def pp_overhead(worker, gpu, ec_collect, t_from=None, t_to=None):
    """Decompose the per-step PP=2 overhead into three measured diffs.

    Formulas (all on the same node clock):
      1. transfer_loss = W.PP1.irecv.end - W.PP0.isend.end
         The stage0->stage1 hidden-state transfer tail. Valid as a cost
         because irecv currently BLOCKS the host: PP1's execute cannot
         proceed until the irecv returns, so the gap between PP0 finishing
         its isend and PP1 finishing its irecv is time PP1 is stalled on
         the transfer (not merely overlapped async work).
      2. bcast_cost = (G.PP1.ppbc.end - G.PP1.exec.cpu_start)
                      - fwd_gpu - samp_gpu
         Exposed wall time of the PP1->PP0 sampled-token broadcast after
         removing the forward and sampling GPU work that precedes it.
      3. pp1_ec = EC.collect.end - G.PP1.ppbc.end
         Idle plumbing between the token being broadcast by PP1 and
         EngineCore collecting the step's output.

    One plumbing cost that lives inside pp1_ec is also shown so it can be
    subtracted from it:
      kvpf   = G.PP1.kv_post_forward cpu time (connector bookkeeping that
               runs inline on the last rank after the broadcast).

    The dominant pp1_ec component is also attributed directly:
      pp0_recv = G.PP0.pp_recv.end - G.PP1.ppbc.end
               PP0 finishes its (short) forward early then blocks in
               pp_handler.receive() for the sampled-token broadcast to land.
               Because sample_tokens aggregates connector output from every
               rank (output_rank=None), EC.collect waits for PP0 too, so this
               exposed receive tail is what pp1_ec is spent on.

    Two idle/wait spans show how early each side blocks before its peer acts:
      pp1_idle = W.PP0.isend.end - W.PP1.irecv.start
               How long PP1 is already parked in the blocking irecv before
               PP0 even finishes sending the hidden states.
      pp0_idle = G.PP1.ppbc.start - G.PP0.pp_recv.start
               How long PP0 is already parked in pp_recv before PP1 begins
               broadcasting the sampled token.
    """
    lo = hms(t_from) if isinstance(t_from, datetime) else t_from
    hi = hms(t_to) if isinstance(t_to, datetime) else t_to
    pp0 = {s: r for (rk, s), r in worker.items() if rk == "PP0"}
    pp0_gpu = {
        s: r
        for (rk, s), r in gpu.items()
        if rk == "PP0" and "pp_recv" in r.get("phases", {})
    }
    w1 = [
        r
        for (rk, _s), r in worker.items()
        if rk == "PP1"
        and r.get("num_tokens") == 1
        and "irecv_post" in r.get("phases", {})
    ]
    w1.sort(key=lambda r: r["phases"]["irecv_post"]["end"])
    w1_t = [r["phases"]["irecv_post"]["end"] for r in w1]
    g1 = [
        r
        for (rk, _s), r in gpu.items()
        if rk == "PP1"
        and r.get("num_tokens") == 1
        and r.get("pre_start") is not None
        and "pp_broadcast" in r.get("phases", {})
    ]
    g1.sort(key=lambda r: r["pre_start"])
    # Only real decode collects: the ntok=0 bubble collects have a near-zero
    # future wait and would otherwise mismatch as the "next" collect. Keep
    # the batch id alongside the end time for per-step attribution.
    coll_pairs = sorted(
        (r["end"], r["batch"])
        for r in ec_collect.values()
        if r.get("end")
        and r["phases"].get("model_output_future_wait", {}).get("cpu_ms", 0) > 1.0
    )
    coll = [t for t, _b in coll_pairs]

    print("# pp_overhead: per decode step (ntok=1)")
    print("#   ECstep            = EngineCore step id (pass to")
    print("#                       --batch-start / --batch-end)")
    print("#   transfer_loss     = W.PP1.irecv.end - W.PP0.isend.end")
    print("#                       (valid: irecv blocks the host)")
    print("#   bcast_cost        = (G.PP1.ppbc.end - G.PP1.exec.cpu_start)")
    print("#                       - fwd_gpu - samp_gpu")
    print("#   pp1_ec            = EC.collect.end - G.PP1.ppbc.end")
    print("#   kvpf              = G.PP1.kv_post_forward cpu (inside pp1_ec)")
    print("#   pp0_recv          = G.PP0.pp_recv.end - G.PP1.ppbc.end")
    print("#                       (PP0 receive tail; what pp1_ec waits on)")
    print("#   pp1_idle          = W.PP0.isend.end - W.PP1.irecv.start")
    print("#                       (PP1 blocked in irecv before PP0 sent)")
    print("#   pp0_idle          = G.PP1.ppbc.start - G.PP0.pp_recv.start")
    print("#                       (PP0 waiting in pp_recv before PP1 bcast)")
    print(
        f"{'ECstep':>7} {'gstep':>5} {'ntok':>5} {'transfer':>9} {'bcast':>8} "
        f"{'pp1_ec':>8} {'kvpf':>7} {'pp0_recv':>8} "
        f"{'pp1_idle':>9} {'pp0_idle':>9}  (ms)"
    )
    tr_s = bc_s = bb_s = kv_s = pr_s = pi_s = po_s = 0.0
    tr_n = bc_n = bb_n = kv_n = pr_n = pi_n = po_n = 0
    for g in g1:
        exec_start = g["pre_start"]
        if lo and hms(exec_start) < lo:
            continue
        if hi and hms(exec_start) > hi:
            continue
        phs = g["phases"]
        ppbc_end = phs["pp_broadcast"]["end"]
        fwd = phs.get("forward", {}).get("gpu_ms") or 0.0
        samp = phs.get("sample", {}).get("gpu_ms") or 0.0
        bcast = _dur_ms(exec_start, ppbc_end) - fwd - samp

        transfer = None
        pp1_idle = None
        i = bisect_left(w1_t, exec_start)
        wrec = w1[i - 1] if i > 0 else None
        if wrec is not None:
            p0 = pp0.get(wrec["step"])
            if p0 and "isend_post" in p0.get("phases", {}):
                transfer = _dur_ms(
                    p0["phases"]["isend_post"]["end"],
                    wrec["phases"]["irecv_post"]["end"],
                )
                pp1_idle = _dur_ms(
                    wrec["phases"]["irecv_post"]["start"],
                    p0["phases"]["isend_post"]["end"],
                )

        pp1_ec = None
        batch = None
        j = bisect_left(coll, ppbc_end)
        if j < len(coll):
            pp1_ec = _dur_ms(ppbc_end, coll[j])
            batch = coll_pairs[j][1]

        kvpf = phs.get("kv_post_forward", {}).get("cpu_ms")

        pp0_recv = None
        pp0_idle = None
        p0g = pp0_gpu.get(g["step"])
        if p0g is not None:
            recv_end = p0g["phases"]["pp_recv"]["end"]
            pp0_recv = _dur_ms(ppbc_end, recv_end)
            pp0_idle = _dur_ms(
                p0g["phases"]["pp_recv"]["start"],
                phs["pp_broadcast"]["start"],
            )

        def f(x):
            return f"{x:.3f}" if x is not None else "-"

        print(
            f"{(batch if batch is not None else '-'):>7} {g['step']:>5} "
            f"{g.get('num_tokens', '-'):>5} "
            f"{f(transfer):>9} {f(bcast):>8} {f(pp1_ec):>8} "
            f"{f(kvpf):>7} {f(pp0_recv):>8} "
            f"{f(pp1_idle):>9} {f(pp0_idle):>9}"
        )
        if transfer is not None:
            tr_s += transfer
            tr_n += 1
        bc_s += bcast
        bc_n += 1
        if pp1_ec is not None:
            bb_s += pp1_ec
            bb_n += 1
        if kvpf is not None:
            kv_s += kvpf
            kv_n += 1
        if pp0_recv is not None:
            pr_s += pp0_recv
            pr_n += 1
        if pp1_idle is not None:
            pi_s += pp1_idle
            pi_n += 1
        if pp0_idle is not None:
            po_s += pp0_idle
            po_n += 1

    def avg(s, n):
        return f"{s / n:.3f}" if n else "-"

    print(
        f"{'avg':>7} {'':>5} {'':>5} {avg(tr_s, tr_n):>9} {avg(bc_s, bc_n):>8} "
        f"{avg(bb_s, bb_n):>8} {avg(kv_s, kv_n):>7} {avg(pr_s, pr_n):>8} "
        f"{avg(pi_s, pi_n):>9} {avg(po_s, po_n):>9}"
    )
    total = (
        (tr_s / tr_n if tr_n else 0)
        + (bc_s / bc_n if bc_n else 0)
        + (bb_s / bb_n if bb_n else 0)
    )
    print(f"# total added overhead/step ~= {total:.3f} ms")


def batch_window_bounds(ec_dispatch, ec_collect, b_start, b_end):
    """Resolve [b_start, b_end] EngineCore batches to a time window.

    Start = dispatch(b_start).start; end = collect(b_end).end. Falls back to
    dispatch(b_end).end if that batch was never collected (e.g. last flush).
    """
    if b_start not in ec_dispatch:
        raise SystemExit(f"batch {b_start} has no EC.dispatch record")
    t_from = ec_dispatch[b_start]["start"]
    if b_end in ec_collect:
        t_to = ec_collect[b_end]["end"]
    elif b_end in ec_dispatch:
        t_to = ec_dispatch[b_end]["end"]
    else:
        raise SystemExit(f"batch {b_end} has no EC.dispatch/collect record")
    return t_from, t_to


def _parse_tp_arg(val):
    """Resolve --tp into (mode, tp_sel). Integer -> ('select', N)."""
    v = str(val).lower()
    if v in ("straggler", "max", "leader", "min", "last"):
        return v, 0
    try:
        return "select", int(v)
    except ValueError:
        raise SystemExit(
            f"--tp: expected an integer or one of "
            f"straggler/max/leader/min/last, got {val!r}"
        ) from None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "logfile",
        help=(
            "Combined log file, or a directory of per-rank files from "
            "VLLM_STEP_TIMING_LOG_DIR (e.g. ./step_timings)."
        ),
    )
    ap.add_argument("--lo", type=int, default=1, help="EngineCore batch low")
    ap.add_argument("--hi", type=int, default=60, help="EngineCore batch high")
    ap.add_argument("--timeline", action="store_true")
    ap.add_argument("--from", dest="t_from", default=None)
    ap.add_argument("--to", dest="t_to", default=None)
    ap.add_argument(
        "--batch-start",
        type=int,
        default=None,
        help="Timeline start = EC.dispatch(batch).start",
    )
    ap.add_argument(
        "--batch-end",
        type=int,
        default=None,
        help="Timeline end = EC.collect(batch).end",
    )
    ap.add_argument(
        "--gantt",
        action="store_true",
        help="Render an ASCII Gantt chart instead of the text timeline",
    )
    ap.add_argument(
        "--gantt-width",
        type=int,
        default=120,
        help="Gantt chart width in columns",
    )
    ap.add_argument(
        "--splits",
        action="store_true",
        help="Gantt: break out irecv/isend/broadcast comms sub-phases",
    )
    ap.add_argument(
        "--pp-overhead",
        action="store_true",
        help="Print per-step PP overhead (transfer/bcast/bubble) + formulas",
    )
    ap.add_argument(
        "--tp",
        default="0",
        help=(
            "TP-rank handling when collapsing per-step records. An integer "
            "selects that TP rank (default 0). Keywords: 'straggler'/'max' = "
            "slowest-finishing rank per (PP,step) (best for collective "
            "phases like pp_broadcast); 'leader'/'min' = fastest; 'last' = "
            "last rank logged (legacy)."
        ),
    )
    args = ap.parse_args()

    mode, tp_sel = _parse_tp_arg(args.tp)
    ec_d, ec_c, worker_all, gpu_all, api = parse(args.logfile)
    worker = reduce_tp(worker_all, mode, tp_sel)
    gpu = reduce_tp(gpu_all, mode, tp_sel)
    events = build_events(ec_d, ec_c, worker, gpu, api)
    if mode == "select":
        print(f"# TP handling: select TP rank {tp_sel}")
    else:
        print(f"# TP handling: {mode} (per PP,step)")
    # Resolve the time window from batch ids (if given) else --from/--to.
    t_from, t_to = args.t_from, args.t_to
    if args.batch_start is not None or args.batch_end is not None:
        b_start = args.batch_start if args.batch_start is not None else args.batch_end
        b_end = args.batch_end if args.batch_end is not None else args.batch_start
        t_from, t_to = batch_window_bounds(ec_d, ec_c, b_start, b_end)
        if args.gantt:
            label = "gantt"
        elif args.pp_overhead:
            label = "pp_overhead"
        else:
            label = "timeline"
        print(
            f"# {label} for EC batches {b_start}-{b_end}: {hms(t_from)} .. {hms(t_to)}"
        )

    if args.gantt:
        gantt(events, t_from, t_to, width=args.gantt_width, splits=args.splits)
    elif args.pp_overhead:
        pp_overhead(worker, gpu, ec_c, t_from, t_to)
    elif args.batch_start is not None or args.batch_end is not None or args.timeline:
        timeline(events, t_from, t_to)
    else:
        join_on_batch(ec_d, ec_c, worker, gpu, api, args.lo, args.hi)


if __name__ == "__main__":
    main()
