"""Calibrate reconnected grain boundaries to the prescribed separation work.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

from ..paths import open_geometry
import json
import os
import numpy as np
from .grain_geometry import gamma_bb

MARGIN = 5.5


def emc_law(m, L, f, seeds, h=0.1, margin=MARGIN):
    """exact per-boundary energy matching on a gb_core.GBModel strip (see module docstring); returns per-boundary info"""
    m.set_law(L)
    X = m.md.X
    W, H, y0, Lz = (m.W, m.H, m.y0, m.Lz)
    xm, ym = (0.5 * (X[m.Ip, 0] + X[m.Jp, 0]), 0.5 * (X[m.Ip, 1] + X[m.Jp, 1]))
    d2 = (xm[:, None] - seeds[None, :, 0]) ** 2 + (ym[:, None] - seeds[None, :, 1]) ** 2
    o = np.argsort(d2, axis=1)[:, :2]
    ba, bb = (np.minimum(o[:, 0], o[:, 1]), np.maximum(o[:, 0], o[:, 1]))
    inwin = (
        (xm > margin)
        & (xm < W - margin)
        & (ym > y0 - H / 2 + margin)
        & (ym < y0 + H / 2 - margin)
    )
    xs = np.arange(0, W, h) + h / 2
    ys = y0 - H / 2 + np.arange(0, H, h) + h / 2
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    Lg = np.argmin(
        (GX[..., None] - seeds[:, 0]) ** 2 + (GY[..., None] - seeds[:, 1]) ** 2, axis=-1
    )
    gw = (
        (GX > margin)
        & (GX < W - margin)
        & (GY > y0 - H / 2 + margin)
        & (GY < y0 + H / 2 - margin)
    )
    wx, wy = (gw[:-1, :] & gw[1:, :], gw[:, :-1] & gw[:, 1:])
    Eb = m.k0 * m.l**2 * L.c[m.sh]
    gb = m.gbb
    info, s2v, lw = ({}, {}, {})
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            sel = gb & (ba == a) & (bb == b)
            if not sel.any():
                continue
            nvec = seeds[b] - seeds[a]
            nvec = nvec / np.linalg.norm(nvec)
            cx = np.sum(
                (
                    (Lg[:-1, :] == a) & (Lg[1:, :] == b)
                    | (Lg[:-1, :] == b) & (Lg[1:, :] == a)
                )
                & wx
            )
            cy = np.sum(
                (
                    (Lg[:, :-1] == a) & (Lg[:, 1:] == b)
                    | (Lg[:, :-1] == b) & (Lg[:, 1:] == a)
                )
                & wy
            )
            Lb = (
                cx * h / abs(nvec[0])
                if abs(nvec[0]) > abs(nvec[1])
                else cy * h / abs(nvec[1])
            )
            n3 = np.array([nvec[0], nvec[1], 0.0])
            tg = (
                (1 - f)
                * 0.5
                * (
                    gamma_bb(m.frames[a].T @ n3, L.E)
                    + gamma_bb(m.frames[b].T @ n3, L.E)
                )
            )
            sw = sel & inwin
            key = "%d-%d" % (a, b)
            info[key] = dict(
                nbonds=int(sel.sum()),
                nbonds_window=int(sw.sum()),
                length_window=float(Lb),
                target2G=float(tg),
            )
            if Lb >= 1.0 and sw.any():
                raw = Eb[sw].sum() / (Lb * Lz)
                s2v[key], lw[key] = (tg / raw, Lb)
                info[key].update(raw2G=float(raw), s2=float(tg / raw))
    s2m = (
        float(np.sum([s2v[k] * lw[k] for k in s2v]) / np.sum(list(lw.values())))
        if s2v
        else 1.0
    )
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            key = "%d-%d" % (a, b)
            if key not in info:
                continue
            s2 = s2v.get(key, s2m)
            if key not in s2v:
                info[key]["s2"] = s2m
                info[key]["s2_from_mean"] = True
            sel = gb & (ba == a) & (bb == b)
            m.l[sel] *= np.sqrt(s2)
            m.E[sel] *= s2
    info["s2_length_weighted_mean"] = s2m
    m.gb_scale = info
    return info


def _bxn_law(m, L, f, seeds, h=0.1):
    info = emc_law(m, L, f, seeds, h)
    for k, v in info.items():
        if isinstance(v, dict):
            print(
                "boundary %s: %d bonds (%d in window), window length %.2f, raw 2G %.4g, target %.4g, g %.4f"
                % (
                    k,
                    v["nbonds"],
                    v["nbonds_window"],
                    v["length_window"],
                    v.get("raw2G", float("nan")),
                    v["target2G"],
                    v["s2"],
                )
            )
    return {k: v for k, v in info.items() if isinstance(v, dict)}


def disc_gb_g_emc(disc, N):
    """per-pair matched multipliers g for the disc's merged model (same particle order as the disc), with a per-boundary
    comparison to the disc's PX calibration"""
    from .grain_geometry import GBModel, law_from_spec
    from .anisotropic_loading import GS

    meta = json.load(open_geometry(disc + ".json"))
    frames = [np.array(F) for F in meta["frames"]]
    seeds = np.array(meta["seeds"])
    shifts = meta.get("shifts")
    if shifts is not None:
        shifts = [None if v is None else tuple(v) for v in shifts]
    assign = lambda P: np.argmin(
        ((P[:, None, :2] - seeds[None, :, :]) ** 2).sum(-1), axis=1
    )
    g = GBModel(
        frames,
        assign,
        kind="strip",
        W=meta["W"],
        H=meta["H"],
        a0=meta["a0"],
        Lz=meta["Lz"],
        shifts=shifts,
    )
    assert g.N == N, (g.N, N)
    L = law_from_spec(meta.get("law", "uber"), g.p, GS["300K"])
    info = emc_law(g, L, float(meta["f"]), seeds)
    old = meta.get("gb_scale", {})
    rows = []
    for k, v in info.items():
        if not isinstance(v, dict):
            continue
        o = old.get(k) if isinstance(old.get(k), dict) else {}
        rows.append(
            (
                k,
                v["nbonds"],
                v["length_window"],
                o.get("length", float("nan")),
                o.get("s2", float("nan")),
                v["s2"],
            )
        )
        print(
            "boundary %-6s %5d bonds  window length %6.2f (PX raster %6.2f)  g: PX %.3f -> matched %.3f%s"
            % (
                k,
                v["nbonds"],
                v["length_window"],
                o.get("length", float("nan")),
                o.get("s2", float("nan")),
                v["s2"],
                "  (mean)" if v.get("s2_from_mean") else "",
            )
        )
    print(
        "PX same-label boundary bonds (scaled by the PX mean): %s; matched length-weighted mean g %.3f"
        % (old.get("n_samelabel_gb_bonds"), info["s2_length_weighted_mean"])
    )
    gpair = (g.l / L.l[g.sh]) ** 2
    out = {}
    for b in np.nonzero(g.gbb & (np.abs(gpair - 1) > 1e-12))[0]:
        out[g.Ip[b], g.Jp[b]] = out[g.Jp[b], g.Ip[b]] = gpair[b]
    return out
