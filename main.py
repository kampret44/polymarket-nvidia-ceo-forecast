#!/usr/bin/env python3
"""Forecast Polymarket 'Next CEO of NVIDIA?' Yes prices, 4h ahead (240 min).

Design:
  * Median: mean-reverts from the checkpoint price x toward an anchor
    (blend of a short recent mean, a long EWMA mean and the sum-normalized
    fair value x / sum(all checkpoint prices)); reversion applies only
    when |anchor - x| exceeds a threshold, so ordinary drift is untouched:
        m(h) = x + (anchor - x) * (1 - exp(-h / tau))
  * Spread, two modes:
      - "normal": q_tau(h) = m(h) + z_tau * sigma * sqrt(h), sigma a robust
        per-market scale from recent per-minute changes (percentile of
        |diff| or trimmed RMS), so rare jumps do not blow it up.
      - "empirical": quantile offsets of historical h-step changes at a
        horizon grid, interpolated in sqrt(h), centred on the median.
  * Sum coherence (optional): shift/scale medians so their sum matches the
    observed checkpoint sum (or 1.0).
  * Quantiles clamped to [0, 1], sorted non-decreasing, rounded to 6 dp.

Stdlib only, deterministic, no network, no randomness.
"""
import json
import math
import os

TAUS = (0.05, 0.25, 0.5, 0.75, 0.95)
Z = (-1.6448536269514722, -0.6744897501960817, 0.0,
     0.6744897501960817, 1.6448536269514722)
# Horizon grid for the empirical spread (minutes).
HGRID = (1, 2, 4, 8, 16, 32, 64, 128, 240)

PARAMS = dict(
    tau=120.0,              # mean-reversion time constant, minutes
    w_recent=0.0,           # anchor weight: mean of last recent_window min
    w_ewma=0.0,             # anchor weight: long EWMA mean
    w_fair=1.0,             # anchor weight: sum-normalized fair value
    sum_target=0.97,        # fair value = x * sum_target / sum(checkpoint)
    recent_window=60,       # minutes for the recent mean
    ewma_halflife=720.0,    # EWMA half-life, minutes
    revert_threshold=0.0,   # only revert if |anchor - x| exceeds this
    spread_mode="normal",   # "normal" | "empirical" | "pooled"
    sigma_mode="trimrms",   # "pct90" | "trimrms" | "fixed"
    sigma_window=1080,      # minutes of recent diffs used for sigma
    sigma_pct=0.90,         # percentile of |diff| for sigma_mode pct
    sigma_pct_scale=2.0,    # multiplier on that percentile
    trim_frac=0.05,         # fraction of largest |diffs| dropped (trimrms)
    sigma_outer_trim=None,  # trim frac for outer (0.05/0.95) quantiles
    jump_boost=0.0,         # outer-sigma boost per unit jump rate
    sigma_fixed=0.001,      # fixed sigma (sigma_mode "fixed")
    sigma_min=0.0004,
    sigma_max=0.01,
    sigma_scale=1.0,
    asymmetric=False,       # separate sigma_up / sigma_down (normal mode)
    small_up_floor=0.0,     # if x < 0.01, floor on sigma_up (asym mode)
    small_up_x=0.01,        # price threshold for small_up_floor
    spread_gamma=0.45,      # spread grows as h**gamma (0.5 = random walk)
    z_outer_scale=1.0,      # multiplier on z at tau 0.05 / 0.95
    z_inner_scale=0.6,      # multiplier on z at tau 0.25 / 0.75
    drift_window=0,         # 0 = no drift; else minutes for recent slope
    drift_cap=0.0001,       # max |drift| per minute
    sum_coherence=None,     # None | "sum" | "one"
    coherence_mode="shift",  # "shift" | "scale"
)


def _ewma(prices, halflife):
    alpha = 1.0 - 0.5 ** (1.0 / halflife)
    e = prices[0]
    for p in prices[1:]:
        e += alpha * (p - e)
    return e


def _pct(sorted_vals, q):
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n == 1:
        return sorted_vals[0]
    pos = q * (n - 1)
    lo = int(pos)
    hi = min(n - 1, lo + 1)
    frac = pos - lo
    return sorted_vals[lo] * (1.0 - frac) + sorted_vals[hi] * frac


def _sigma(diffs, p):
    """Robust per-market sigma (per sqrt-minute) from recent diffs."""
    n = len(diffs)
    if n == 0:
        return None
    mode = p["sigma_mode"]
    if mode == "fixed":
        return p["sigma_fixed"]
    if mode == "pct90":
        absd = sorted(abs(d) for d in diffs)
        return _pct(absd, p["sigma_pct"]) * p["sigma_pct_scale"]
    if mode == "trimrms":
        absd = sorted(abs(d) for d in diffs)
        keep = max(1, int(n * (1.0 - p["trim_frac"])))
        thr = absd[keep - 1]
        if not p["asymmetric"]:
            s = 0.0
            cnt = 0
            for d in diffs:
                c = d if -thr <= d <= thr else (thr if d > 0 else -thr)
                s += c * c
                cnt += 1
            return math.sqrt(s / cnt)
        su = 0.0
        nu = 0
        sd = 0.0
        nd = 0
        for d in diffs:
            c = d if -thr <= d <= thr else (thr if d > 0 else -thr)
            if c > 0:
                su += c * c
                nu += 1
            elif c < 0:
                sd += c * c
                nd += 1
        up = math.sqrt(su / nu) if nu else 0.0
        dn = math.sqrt(sd / nd) if nd else 0.0
        return (up, dn)
    raise ValueError(mode)


def _empirical_offsets(prices, p):
    """Per-horizon-grid empirical quantile offsets of h-step changes.

    Returns {h: [off_05, off_25, off_50, off_75, off_95]} where offsets are
    centred on the empirical median change (so the median stays the
    reversion median). None when history is too short.
    """
    n = len(prices)
    out = {}
    for h in HGRID:
        if n <= h + 20:
            continue
        ch = sorted(prices[j] - prices[j - h] for j in range(h, n))
        med = _pct(ch, 0.5)
        out[h] = [_pct(ch, t) - med for t in TAUS]
    return out or None


def _pooled_offsets(markets, sigmas, p):
    """Pooled empirical quantiles of standardized h-step changes.

    For each grid horizon h, collect (price[t]-price[t-h])/(sigma_i*sqrt(h))
    over all markets and times; return {h: [q05..q95]} centred on the
    pooled median change. Captures the true heavy-tail shape (mostly-zero
    moves plus jumps) far more stably than per-market empirical quantiles.
    """
    out = {}
    smin = p["sigma_min"]
    for h in HGRID:
        vals = []
        sh = math.sqrt(h)
        for i, m in enumerate(markets):
            prices = m["prices"]
            n = len(prices)
            if n <= h + 20:
                continue
            s = max(smin, sigmas[i] if not isinstance(sigmas[i], tuple)
                    else sigmas[i][0])
            denom = s * sh
            # step 2 keeps the sort cheap; changes overlap heavily anyway
            for t in range(h, n, 2):
                vals.append((prices[t] - prices[t - h]) / denom)
        if len(vals) < 200:
            continue
        vals.sort()
        med = _pct(vals, 0.5)
        out[h] = [_pct(vals, t) - med for t in TAUS]
    return out or None


def _interp_offsets(offsets, h):
    """Interpolate empirical offsets for horizon h (linear in sqrt(h))."""
    keys = sorted(offsets)
    if h <= keys[0]:
        return offsets[keys[0]]
    if h >= keys[-1]:
        return offsets[keys[-1]]
    sh = math.sqrt(h)
    for a, b in zip(keys, keys[1:]):
        if a <= h <= b:
            sa, sb = math.sqrt(a), math.sqrt(b)
            f = (sh - sa) / (sb - sa) if sb > sa else 0.0
            oa, ob = offsets[a], offsets[b]
            return [oa[k] + f * (ob[k] - oa[k]) for k in range(5)]
    return offsets[keys[-1]]


def forecast_case(case, params=None):
    p = dict(PARAMS)
    if params:
        p.update(params)
    markets = case["markets"]
    n = len(markets)
    xs = [m["prices"][-1] for m in markets]
    total = sum(xs)
    fair = [x * p["sum_target"] / total if total > 0 else x for x in xs]

    anchors = []
    sigmas = []
    sigmas_outer = []
    emps = []
    for i, m in enumerate(markets):
        prices = m["prices"]
        wr = p["w_recent"]
        we = p["w_ewma"]
        wf = p["w_fair"]
        anchor = 0.0
        if wr:
            k = p["recent_window"]
            seg = prices[-k:] if len(prices) > k else prices
            anchor += wr * (sum(seg) / len(seg))
        if we:
            anchor += we * _ewma(prices, p["ewma_halflife"])
        if wf:
            anchor += wf * fair[i]
        if wr + we + wf == 0.0:
            anchor = xs[i]
        anchors.append(anchor)
        w = p["sigma_window"]
        start = max(1, len(prices) - w)
        diffs = [prices[j] - prices[j - 1] for j in range(start, len(prices))]
        sigmas.append(_sigma(diffs, p))
        if p["sigma_outer_trim"] is not None:
            q2 = dict(p)
            q2["trim_frac"] = p["sigma_outer_trim"]
            so = _sigma(diffs, q2)
            if p["jump_boost"]:
                jr = sum(1 for d in diffs if abs(d) > 0.01) / max(1, len(diffs))
                so = so * (1.0 + p["jump_boost"] * jr)
            sigmas_outer.append(so)
        else:
            sigmas_outer.append(None)
        emps.append(_empirical_offsets(prices, p)
                    if p["spread_mode"] == "empirical" else None)

    flat = sorted(s for s in sigmas if s is not None)
    med_sig = flat[len(flat) // 2] if flat else p["sigma_fixed"]
    sigmas = [s if s is not None else med_sig for s in sigmas]
    sigmas_outer = [s if s is not None else sigmas[i]
                    for i, s in enumerate(sigmas_outer)]
    pooled = (_pooled_offsets(markets, sigmas, p)
              if p["spread_mode"] == "pooled" else None)

    tau = p["tau"]
    smin = p["sigma_min"]
    smax = p["sigma_max"]
    rev = [1.0 - math.exp(-h / tau) for h in range(1, 241)]
    sqrth = [math.sqrt(h) for h in range(1, 241)]
    hg = [(h + 1) ** p["spread_gamma"] for h in range(240)]
    zs = (Z[0] * p["z_outer_scale"], Z[1] * p["z_inner_scale"], 0.0,
          Z[3] * p["z_inner_scale"], Z[4] * p["z_outer_scale"])
    thr = p["revert_threshold"]

    med = [[0.0] * n for _ in range(240)]
    for i in range(n):
        x = xs[i]
        a = anchors[i]
        d = a - x
        if abs(d) < thr:
            d = 0.0
        slope = 0.0
        if p["drift_window"]:
            kw = p["drift_window"]
            pr = markets[i]["prices"]
            k = min(kw, len(pr) - 1)
            if k > 0:
                slope = (x - pr[-1 - k]) / k
                cap = p["drift_cap"]
                slope = max(-cap, min(cap, slope))
        for h in range(240):
            med[h][i] = x + d * rev[h] + slope * (h + 1)

    if p["sum_coherence"] is not None:
        target = total if p["sum_coherence"] == "sum" else 1.0
        for h in range(240):
            s = sum(med[h])
            if p["coherence_mode"] == "shift":
                delta = (target - s) / n
                if delta:
                    row = med[h]
                    for i in range(n):
                        row[i] += delta
            elif s > 0:
                f = target / s
                row = med[h]
                for i in range(n):
                    row[i] *= f

    out = {}
    for i, m in enumerate(markets):
        rows = []
        emp = emps[i]
        if p["spread_mode"] == "empirical" and emp is not None:
            for h in range(240):
                mh = med[h][i]
                offs = _interp_offsets(emp, h + 1)
                vals = [mh + o for o in offs]
                vals = [0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
                        for v in vals]
                vals.sort()
                rows.append([round(v, 6) for v in vals])
        else:
            s = sigmas[i]
            if p["spread_mode"] == "pooled" and pooled is not None:
                sv = min(smax, max(smin, s)) * p["sigma_scale"]
                for h in range(240):
                    mh = med[h][i]
                    offs = _interp_offsets(pooled, h + 1)
                    sc = sv * sqrth[h]
                    vals = [mh + o * sc for o in offs]
                    vals = [0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
                            for v in vals]
                    vals.sort()
                    rows.append([round(v, 6) for v in vals])
                out[m["id"]] = rows
                continue
            if p["asymmetric"]:
                su = min(smax, max(smin, s[0])) * p["sigma_scale"]
                sd = min(smax, max(smin, s[1])) * p["sigma_scale"]
                if p["small_up_floor"] and xs[i] < p["small_up_x"]:
                    su = max(su, p["small_up_floor"])
                for h in range(240):
                    mh = med[h][i]
                    vals = [mh + (su if z >= 0 else sd) * z * hg[h]
                            for z in zs]
                    vals = [0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
                            for v in vals]
                    vals.sort()
                    rows.append([round(v, 6) for v in vals])
                out[m["id"]] = rows
                continue
            sv = min(smax, max(smin, s)) * p["sigma_scale"]
            svo = min(smax, max(smin, sigmas_outer[i])) * p["sigma_scale"]
            for h in range(240):
                mh = med[h][i]
                spi = sv * hg[h]
                spo = svo * hg[h]
                vals = [mh + zs[0] * spo, mh + zs[1] * spi, mh,
                        mh + zs[3] * spi, mh + zs[4] * spo]
                vals = [0.0 if v < 0.0 else (1.0 if v > 1.0 else v)
                        for v in vals]
                vals.sort()
                rows.append([round(v, 6) for v in vals])
        out[m["id"]] = rows
    return out


def main():
    in_dir = os.environ.get("MAKEFUN_INPUT_DIR", "/input/")
    out_dir = os.environ.get("MAKEFUN_OUTPUT_DIR", "/output/")
    os.makedirs(out_dir, exist_ok=True)
    names = sorted(f for f in os.listdir(in_dir) if f.endswith(".in"))
    for name in names:
        with open(os.path.join(in_dir, name)) as f:
            case = json.load(f)
        forecasts = forecast_case(case)
        out = {"forecasts": [{"id": mid, "q": rows}
                             for mid, rows in forecasts.items()]}
        out_name = name[:-3] + ".out" if name.endswith(".in") else name + ".out"
        with open(os.path.join(out_dir, out_name), "w") as f:
            json.dump(out, f, separators=(",", ":"))


if __name__ == "__main__":
    main()
