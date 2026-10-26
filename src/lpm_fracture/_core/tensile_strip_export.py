"""Tensile strip export.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

from ..paths import open_geometry
import json
import os
import numpy as np
from . import model_export as ge


def case_pxu_t(kind, out, disc):
    from .grain_geometry import GBModel

    meta = json.load(open_geometry(disc + ".json"))
    assert (
        meta.get("transparent") and int(meta.get("kbc", 0)) == 0
    ), "expects a transparent clamped-strip disc"
    frames = [np.array(F) for F in meta["frames"]]
    seeds = np.array(meta["seeds"])
    W, H, a0, Lz = (meta["W"], meta["H"], meta["a0"], meta["Lz"])
    shifts = meta.get("shifts")
    if shifts is not None:
        shifts = [None if v is None else tuple(v) for v in shifts]
    assign = lambda P: np.argmin(
        ((P[:, None, :2] - seeds[None, :, :]) ** 2).sum(-1), axis=1
    )
    grip, prot = (1.2, 4.0)
    s_c = 0.0746
    p = ge.params(ge.NEP["C11"], ge.NEP["C12"], ge.NEP["C44"], a=1.0, mapping="paper")
    kg = 1e-07 * p["Kn"][0]
    if kind == "naive":
        g = GBModel(frames, assign, kind="strip", W=W, H=H, a0=a0, Lz=Lz, shifts=shifts)
        assert g.N == meta["N"] and g.P == meta["P"], (g.N, g.P)
        X, y0 = (g.md.X, g.y0)
        m = ge.naive_lpm(g.md, g.p)
        pair = m.uid
        flag = g.protb[pair].astype(np.uint8) << 1 | g.cut[pair].astype(np.uint8) << 2
        fixed = np.zeros(3 * len(X), bool)
        fixed[g.fixed] = True
        ubc = g.uK.copy()
        kn, T = (m.kn0, g.p["T"])
        gg = ge.disc_gb_g(disc, len(X))
        gm = np.array([gg.get((a, b), 1.0) for a, b in zip(m.i, m.j)])
        if shifts is None:
            gm[:] = 1.0
        print(
            "merged model: %d particles; GB multipliers: %d directed bonds, g %.3f-%.3f"
            % (
                len(X),
                int((gm != 1).sum()),
                gm[gm != 1].min() if (gm != 1).any() else 1.0,
                gm.max(),
            )
        )
    else:
        y0 = meta["y0"]
        lo, hi = ((0.0, y0 - H / 2), (W, y0 + H / 2))
        X, gid = ge.build_grains(frames, assign, lo, hi, Lz, shifts=shifts)
        m = ge.GhostLPM(
            X, gid, frames, assign, ge.NEP, Lz, lo, hi, crack_aware=False, interp="mls"
        )
        print(
            "ghost model: %d particles, %d bonds, %d ghost, %d MLS fallbacks"
            % (m.N, m.nb, int(m.ghost.sum()), m.nfallback)
        )
        wopt = os.environ.get("GH_WOPT")
        if wopt:
            from .weight_optimization import (
                optimise_weights_gen,
                set_weights,
                poly_facet_fields,
            )
            from .elasticity import cubic_C

            flds, nfac = poly_facet_fields(
                X,
                gid,
                seeds,
                frames,
                cubic_C(ge.NEP["C11"], ge.NEP["C12"], ge.NEP["C44"]),
            )
            dbc0 = np.minimum.reduce([X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]])
            wd, winfo = optimise_weights_gen(
                m,
                (dbc0 >= grip + 1.4) & (X[:, 0] > 2.6) & (X[:, 0] < W - 2.6),
                flds,
                mu=float(wopt),
                per_owner=True,
            )
            set_weights(m, wd)
            print(
                "optimised weights (mu %s, %d facets): %s"
                % (
                    wopt,
                    nfac,
                    {k: winfo[k] for k in ("resid_mls", "resid_opt", "w_min", "w_max")},
                )
            )
        pair = m.pair
        top, bot = (X[:, 1] > y0 + H / 2 - grip, X[:, 1] < y0 - H / 2 + grip)
        protp = (X[:, 1] > y0 + H / 2 - grip - prot) | (
            X[:, 1] < y0 - H / 2 + grip + prot
        )
        fixed = np.repeat(top | bot, 3)
        ubc = np.zeros((len(X), 3))
        ubc[top, 1], ubc[bot, 1] = (0.5, -0.5)
        ubc = ubc.ravel()
        site = X[m.i] - m.n * m.d0[:, None]
        xm = 0.5 * (X[m.i, 0] + site[:, 0])
        cut = ((X[m.i, 1] - y0) * (site[:, 1] - y0) < 0) & (xm < a0)
        cutp = np.zeros(m.nb, bool)
        cutp[pair[cut]] = True
        G = m.Gj.tocsr()
        far = np.array(
            [
                G.indices[G.indptr[b] : G.indptr[b + 1]][
                    np.argmax(G.data[G.indptr[b] : G.indptr[b + 1]])
                ]
                // 2
                for b in range(m.nb)
            ]
        )
        pr = protp[m.i] | protp[far]
        prp = np.zeros(m.nb, bool)
        prp[pair[pr]] = True
        flag = (
            m.ghost.astype(np.uint8)
            | prp[pair].astype(np.uint8) << 1
            | cutp[pair].astype(np.uint8) << 2
        )
        kn, T = (m.Kn, m.T)
        gm = np.ones(len(m.i))
        np.save(out[:-4] + "_gid.npy", gid)
    coh = (flag & 2 == 0) & (flag & 4 == 0)
    extra = {
        "hdr": [meta["l_h"][0], meta["l_h"][1], y0, H, W, a0],
        "coh": coh,
        "gmul": gm,
    }
    ge.write(
        out,
        X,
        fixed,
        ubc,
        m,
        kn,
        T,
        flag,
        np.ones(len(m.i)),
        pair,
        s_c=s_c,
        tie=1e-06,
        kg=kg,
        Lz=Lz,
        extra=extra,
    )
    json.dump(
        {
            "disc": disc,
            "kind": kind,
            "wopt": os.environ.get("GH_WOPT"),
            "N": int(len(X)),
            "shifts": meta.get("shifts"),
            "seeds": meta["seeds"],
            "W": W,
            "H": H,
            "a0": a0,
            "Lz": Lz,
            "y0": float(y0),
        },
        open(out[:-4] + ".json", "w"),
        indent=1,
    )
