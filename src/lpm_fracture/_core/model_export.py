"""Export rotated-grain lattice models to the cohesive solver binary format.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

from ..paths import open_geometry
import json
import os
import numpy as np
from .bcc_lattice import params
from .ghost_lattice import GhostLPM, build_grains
from .linear_energy import LPM3D

NEP = dict(C11=250.7, C12=123.8, C44=98.7)


def supports_of(m):
    """CSR support lists (far end) of every bond: ghost model from B0 minus the owner entry, naive = (j, 1)."""
    if isinstance(m, GhostLPM):
        G = m.Gj.tocsr()
        G.sort_indices()
        sidx = (G.indices // 2).astype(np.int32)
        return (G.indptr.astype(np.int64), sidx, G.data.astype(float))
    nb = m.nb
    return (np.arange(nb + 1, dtype=np.int64), m.j.astype(np.int32), np.ones(nb))


def write(path, X, fixed, ubc, m, kn, T, flag, xi, pair, s_c, tie, kg, Lz, extra=None):
    sptr, sidx, sw = supports_of(m)
    nb = len(m.i)
    with open(path, "wb") as f:
        np.array(
            [len(X), nb, len(sidx), 1 if extra is None else 2, 0, 0, 0, 0], np.int64
        ).tofile(f)
        if extra is None:
            np.array([T, s_c, tie, kg, Lz, 0, 0, 0], float).tofile(f)
        else:
            hd = np.zeros(16)
            hd[:11] = [T, s_c, tie, kg, Lz] + list(extra["hdr"])
            hd.tofile(f)
        np.ascontiguousarray(X, float).tofile(f)
        fixed.astype(np.uint8).tofile(f)
        ubc.astype(float).tofile(f)
        m.i.astype(np.int32).tofile(f)
        m.s.astype(np.int32).tofile(f)
        pair.astype(np.int32).tofile(f)
        flag.astype(np.uint8).tofile(f)
        np.ascontiguousarray(m.n, float).tofile(f)
        m.d0.astype(float).tofile(f)
        kn.astype(float).tofile(f)
        xi.astype(float).tofile(f)
        sptr.tofile(f)
        sidx.tofile(f)
        sw.tofile(f)
        if extra is not None:
            extra["coh"].astype(np.uint8).tofile(f)
            extra["gmul"].astype(float).tofile(f)
    print(
        "%s: N %d bonds %d (ghost %d, protected %d, pre-cut %d) supports %d, T %.6g s_c %.4g kg %.3g"
        % (
            path,
            len(X),
            nb,
            int((flag & 1).astype(bool).sum()),
            int((flag & 2).astype(bool).sum()),
            int((flag & 4).astype(bool).sum()),
            len(sidx),
            T,
            s_c,
            kg,
        )
    )


def naive_lpm(bm, p):
    m = LPM3D(bm, p["Kn"], p["T"])
    m.s = bm.sh
    return m


def disc_gb_g(disc, N):
    """per-pair GB separation-energy multipliers g of the PX disc file (J04/PX particle order)."""
    lines = open_geometry(disc + ".txt").read().splitlines()
    k = 2 + N
    npre = int(lines[k])
    k += 1 + npre
    ngb = int(lines[k].split()[0])
    k += 1
    out = {}
    for l in lines[k : k + ngb]:
        i, j, g = l.split()
        out[int(i), int(j)] = out[int(j), int(i)] = float(g)
    return out


def case_px(kind, out, disc="PX_s1_f0", interp="mls", uber=False):
    from .grain_geometry import GBModel

    meta = json.load(open_geometry(disc + ".json"))
    frames = [np.array(F) for F in meta["frames"]]
    seeds = np.array(meta["seeds"])
    W, H, a0, Lz = (meta["W"], meta["H"], meta["a0"], meta["Lz"])
    assign = lambda P: np.argmin(
        ((P[:, None, :2] - seeds[None, :, :]) ** 2).sum(-1), axis=1
    )
    grip, prot = (1.2, 4.0)
    s_c = 0.0746
    p = params(NEP["C11"], NEP["C12"], NEP["C44"], a=1.0, mapping="paper")
    kg = 1e-07 * p["Kn"][0]
    if kind == "naive":
        g = GBModel(frames, assign, kind="strip", W=W, H=H, a0=a0, Lz=Lz)
        assert g.N == meta["N"] and g.P == meta["P"], (g.N, g.P)
        X, y0 = (g.md.X, g.y0)
        m = naive_lpm(g.md, g.p)
        pair = m.uid
        flag = g.protb[pair].astype(np.uint8) << 1 | g.cut[pair].astype(np.uint8) << 2
        fixed = np.zeros(3 * len(X), bool)
        fixed[g.fixed] = True
        ubc = g.uK.copy()
        kn, T = (m.kn0, g.p["T"])
    else:
        y0 = meta["y0"]
        lo, hi = ((0.0, y0 - H / 2), (W, y0 + H / 2))
        X, gid = build_grains(frames, assign, lo, hi, Lz)
        m = GhostLPM(
            X, gid, frames, assign, NEP, Lz, lo, hi, crack_aware=False, interp=interp
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
                X, gid, seeds, frames, cubic_C(NEP["C11"], NEP["C12"], NEP["C44"])
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
        np.save(out[:-4] + "_gid.npy", gid)
    extra = None
    if uber:
        gm = np.ones(len(m.i))
        if kind == "naive":
            gg = disc_gb_g(disc, len(X))
            gm = np.array([gg.get((a, b), 1.0) for a, b in zip(m.i, m.j)])
            print(
                "J04/PX GB multipliers: %d directed bonds, g %.3f-%.3f"
                % (int((gm != 1).sum()), gm[gm != 1].min(), gm.max())
            )
        coh = (flag & 2 == 0) & (flag & 4 == 0)
        extra = {
            "hdr": [meta["l_h"][0], meta["l_h"][1], y0, H, W, a0],
            "coh": coh,
            "gmul": gm,
        }
    write(
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


def read_disc_particles(disc):
    lines = open_geometry(disc + ".txt").read().splitlines()
    N = int(lines[0].split()[0])
    rows = np.array([[float(v) for v in l.split()] for l in lines[2 : 2 + N]])
    kf = [l for l in lines if l.startswith("KF")]
    return (rows, np.array([float(v) for v in kf[0].split()[1:]]) if kf else None)


def case_pxk(kind, out, disc="PXK_s1_f0", interp="mls"):
    """plmode 8 UBER (version 2 binary) + sidecar <out>.kf = 'KIG a0 y0' and the 30 Stroh numbers of the disc KF line.
    J04/PX: geometry, boundary bits, protection and GB multipliers g exactly as the disc (fork drv_px input); ghost: the
    same rules (px_export kbc=1) on the ghost lattice, ghost-bond multiplier g = 1 - f (f = 0: plain bulk law).
    """
    from .grain_geometry import GBModel

    meta = json.load(open_geometry(disc + ".json"))
    frames = [np.array(F) for F in meta["frames"]]
    seeds = np.array(meta["seeds"])
    W, H, a0, Lz, f = (meta["W"], meta["H"], meta["a0"], meta["Lz"], meta["f"])
    shifts = meta.get("shifts")
    if shifts is not None:
        shifts = [None if v is None else tuple(v) for v in shifts]
    a0x = float(os.environ.get("GH_A0", a0))
    assign = lambda P: np.argmin(
        ((P[:, None, :2] - seeds[None, :, :]) ** 2).sum(-1), axis=1
    )
    rows, kfnum = read_disc_particles(disc)
    p = params(NEP["C11"], NEP["C12"], NEP["C44"], a=1.0, mapping="paper")
    kg = 1e-07 * p["Kn"][0]
    y0 = meta["y0"]
    if kind == "naive":
        g = GBModel(frames, assign, kind="strip", W=W, H=H, a0=a0, Lz=Lz, shifts=shifts)
        X = g.md.X
        assert np.allclose(X, rows[:, :3]), "particle order differs from the disc"
        m = naive_lpm(g.md, g.p)
        pair = m.uid
        bits, protp = (rows[:, 3].astype(int), rows[:, 4].astype(bool))
        fixed = np.stack([bits >> a & 1 for a in range(3)], 1).ravel().astype(bool)
        prb = protp[m.i] | protp[m.j]
        cutn = g.cross & (g.xm < a0x)
        flag = prb.astype(np.uint8) << 1 | cutn[pair].astype(np.uint8) << 2
        kn, T = (m.kn0, g.p["T"])
        gg = disc_gb_g(disc, len(X))
        gm = np.array([gg.get((a, b), 1.0) for a, b in zip(m.i, m.j)])
        if meta.get("transparent") and shifts is None:
            gm[:] = 1.0
        fgb = os.environ.get("GH_FGB")
        if fgb:
            assert f == 0.0, "use the f = 0 disc with GH_FGB"
            pf = json.load(open(fgb))["pair_f"]
            lab = assign(X)
            sel = np.nonzero(gm != 1.0)[0]
            miss = 0
            for b_ in sel:
                la, lb = (lab[m.i[b_]], lab[m.j[b_]])
                key = "%d-%d" % (min(la, lb), max(la, lb))
                if key in pf:
                    gm[b_] *= 1.0 - pf[key]
                else:
                    miss += 1
            print(
                "per-boundary f from %s: %d J04 GB bonds, %d without an entry"
                % (os.path.basename(fgb), len(sel), miss)
            )
    else:
        lo, hi = ((0.0, y0 - H / 2), (W, y0 + H / 2))
        X, gid = build_grains(frames, assign, lo, hi, Lz, shifts=shifts)
        m = GhostLPM(
            X, gid, frames, assign, NEP, Lz, lo, hi, crack_aware=False, interp=interp
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
                X, gid, seeds, frames, cubic_C(NEP["C11"], NEP["C12"], NEP["C44"])
            )
            dbc0 = np.minimum.reduce(
                [W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
            )
            wd, winfo = optimise_weights_gen(
                m, (dbc0 >= 2.6) & (X[:, 0] > 2.6), flds, mu=float(wopt), per_owner=True
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
        dmin = np.minimum.reduce(
            [X[:, 0], W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
        )
        dbc = np.minimum.reduce(
            [W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
        )
        fixed = np.repeat(dbc < 1.5, 3)
        protp = dmin < 5.5
        site = X[m.i] - m.n * m.d0[:, None]
        xm = 0.5 * (X[m.i, 0] + site[:, 0])
        cut = ((X[m.i, 1] - y0) * (site[:, 1] - y0) < 0) & (xm < a0x)
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
        gm = np.where(m.ghost, 1.0 - f, 1.0)
        fgb = os.environ.get("GH_FGB")
        if fgb:
            pf = json.load(open(fgb))["pair_f"]
            ga, gb_ = (gid[m.i], m.sgrain)
            miss = 0
            for b_ in np.nonzero(m.ghost)[0]:
                key = "%d-%d" % (min(ga[b_], gb_[b_]), max(ga[b_], gb_[b_]))
                if key in pf:
                    gm[b_] = 1.0 - pf[key]
                else:
                    miss += 1
            print(
                "per-boundary f from %s: %d ghost bonds, %d without an entry (kept 1 - f_disc)"
                % (os.path.basename(fgb), int(m.ghost.sum()), miss)
            )
        np.save(out[:-4] + "_gid.npy", gid)
    print(
        "GB multipliers: %d directed bonds with g != 1 (range %.3f-%.3f), f = %g"
        % (int((gm != 1).sum()), gm.min(), gm.max(), f)
    )
    gj = os.environ.get("GH_GJIT")
    if gj:
        up = np.random.default_rng(int(os.environ.get("GH_SEED", "1"))).uniform(
            -1.0, 1.0, int(pair.max()) + 1
        )
        gm = gm * (1.0 + float(gj) * up[pair])
        print(
            "strength jitter +-%s (seed %s): g %.4f-%.4f"
            % (gj, os.environ.get("GH_SEED", "1"), gm.min(), gm.max())
        )
    coh = (flag & 2 == 0) & (flag & 4 == 0)
    extra = {
        "hdr": [meta["l_h"][0], meta["l_h"][1], y0, H, W, a0x],
        "coh": coh,
        "gmul": gm,
    }
    write(
        out,
        X,
        fixed,
        np.zeros(3 * len(X)),
        m,
        kn,
        T,
        flag,
        np.ones(len(m.i)),
        pair,
        s_c=0.0746,
        tie=1e-06,
        kg=kg,
        Lz=Lz,
        extra=extra,
    )
    with open(out[:-4] + ".kf", "w") as fo:
        fo.write("%.17g %.17g %.17g\n" % (meta["K0_GPa_sqrt_h"], a0x, y0))
        fo.write(" ".join(("%.17g" % v for v in kfnum)) + "\n")
    print(
        "K field: KIG %.6g, %d boundary particles"
        % (meta["K0_GPa_sqrt_h"], int(fixed.reshape(-1, 3)[:, 0].sum()))
    )


TRIPLES = {5: (3, 4), 13: (5, 12), 17: (8, 15), 25: (7, 24), 29: (20, 21)}
SHIFT_BX = (0.37, 0.61, 0.23)


def l_h_for(r, meta):
    """UBER lengths (units h) of the two shells: the disc's NEP law, or the PX sensitivity law 'aniso:r'
    (2gamma{110} = r 2gamma{100}, gamma{100} = NEP; px_export.py with s = 10)."""
    if not r or float(r) <= 0:
        return list(meta["l_h"])
    from .cleavage_calibration import calib_bond
    from .anisotropic_loading import NEP_C, GS, A_NEP

    E, l = calib_bond(
        params(*NEP_C),
        {"100": GS["300K"]["100"], "110": float(r) * GS["300K"]["100"]},
        a_h=10 * A_NEP,
    )
    assert np.all(E > 0), E
    return [float(l[0]), float(l[1])]


def case_bx(
    out,
    mode="twist",
    sigma=5,
    f=0.0,
    Lz=None,
    W=56.0,
    H=36.0,
    a0=12.0,
    xg=16.0,
    interp="mls",
    disc="PXK_s1_f0",
    r=0.0,
    shift=None,
    hmls=0.9,
    dpar=2.0,
    gbang=0.0,
):
    """gbang (deg, default 0): the boundary through (xg, y0) is rotated about z by gbang (normal (cos, sin, 0)); 0 = the
    vertical boundary x = xg of the original crossing tests"""
    W, H, a0, xg, gbang = (float(W), float(H), float(a0), float(xg), float(gbang))
    cga, sga = (np.cos(np.radians(gbang)), np.sin(np.radians(gbang)))
    from .grain_geometry import R_A, rotx, rotz, grain_frame, crack_y0

    sigma = int(sigma)
    m_, n_ = TRIPLES[sigma]
    ang = np.degrees(np.arctan2(m_, n_)) if mode not in ("shift", "shiftpar") else 0.0
    Lz = float(Lz) if Lz else float(sigma) if mode == "twist" else 5.0
    shv = tuple((float(v) for v in shift.split(","))) if shift else SHIFT_BX
    shifts = [None, shv] if mode in ("shift", "shiftpar") else None
    if mode == "twist":
        frames = [R_A, grain_frame(rotx(ang))]
    elif mode == "tilt":
        frames = [R_A, grain_frame(rotz(ang))]
    else:
        frames = [R_A, R_A]
    y0_ = crack_y0(R_A, float(Lz) if Lz else float(sigma) if mode == "twist" else 5.0)
    assign = (
        (lambda P: ((P[:, 0] - xg) * cga + (P[:, 1] - y0_) * sga >= 0).astype(int))
        if mode != "single"
        else lambda P: np.zeros(len(P), int)
    )
    if mode == "shiftpar":
        ypl = crack_y0(R_A, Lz) + float(dpar)
        assign = lambda P: (P[:, 1] >= ypl).astype(int)
    meta = json.load(open_geometry(disc + ".json"))
    _, kfnum = read_disc_particles(disc)
    y0 = crack_y0(R_A, Lz)
    p = params(NEP["C11"], NEP["C12"], NEP["C44"], a=1.0, mapping="paper")
    kg = 1e-07 * p["Kn"][0]
    lo, hi = ((0.0, y0 - H / 2), (W, y0 + H / 2))
    X, gid = build_grains(frames, assign, lo, hi, Lz, shifts=shifts)
    X = np.round(X, 10)
    X[:, 2] = np.mod(X[:, 2], Lz)
    X[X[:, 2] >= Lz - 1e-09, 2] = 0.0
    m = GhostLPM(
        X,
        gid,
        frames,
        assign,
        NEP,
        Lz,
        lo,
        hi,
        crack_aware=False,
        interp=interp,
        hmls=float(hmls),
    )
    wopt = os.environ.get("GH_WOPT")
    if wopt and mode != "single":
        from .weight_optimization import (
            optimise_weights_gen,
            set_weights,
            exact_bicrystal_n,
        )
        from .elasticity import cubic_C, rotate_C

        C0 = cubic_C(NEP["C11"], NEP["C12"], NEP["C44"])
        CA, CB = (rotate_C(C0, frames[0]), rotate_C(C0, frames[1]))
        par = mode == "shiftpar"
        x0 = np.array([0.0, ypl, 0.0]) if par else np.array([xg, y0, 0.0])
        nrm = [0.0, 1.0, 0.0] if par else [cga, sga, 0.0]
        flds = []
        for E in (
            np.diag([1.0, 0, 0]),
            np.diag([0, 1.0, 0]),
            np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
        ):
            HA, HB = exact_bicrystal_n(E, CA, CB, nrm)
            flds.append(
                lambda P, HA=HA, HB=HB: np.where(
                    (
                        P[:, 1] < ypl
                        if par
                        else (P[:, 0] - xg) * cga + (P[:, 1] - y0) * sga < 0
                    )[:, None],
                    (P - x0) @ HA.T,
                    (P - x0) @ HB.T,
                )
            )
        dbc0 = np.minimum.reduce(
            [W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
        )
        wd, winfo = optimise_weights_gen(
            m, (dbc0 >= 2.6) & (X[:, 0] > 2.6), flds, mu=float(wopt)
        )
        set_weights(m, wd)
    inner = np.abs(X[m.i, 0] - xg) > 3.0
    print(
        "bx %s Sigma %d angle %.3f deg Lz %g: %d particles, %d bonds, %d ghost (%d farther than 3 h from x = xg: lattice periodicity check), %d MLS fallbacks, y0 %.3f"
        % (
            mode,
            sigma,
            ang,
            Lz,
            m.N,
            m.nb,
            int(m.ghost.sum()),
            int((m.ghost & inner).sum()),
            m.nfallback,
            y0,
        )
    )
    pair = m.pair
    dmin = np.minimum.reduce(
        [X[:, 0], W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
    )
    dbc = np.minimum.reduce([W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]])
    fixed = np.repeat(dbc < 1.5, 3)
    protp = dmin < 5.5
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
    gm = np.where(m.ghost, 1.0 - float(f), 1.0)
    coh = (flag & 2 == 0) & (flag & 4 == 0)
    lh = l_h_for(r, meta)
    extra = {"hdr": [lh[0], lh[1], y0, H, W, a0], "coh": coh, "gmul": gm}
    write(
        out,
        X,
        fixed,
        np.zeros(3 * len(X)),
        m,
        m.Kn,
        m.T,
        flag,
        np.ones(len(m.i)),
        pair,
        s_c=0.0746,
        tie=1e-06,
        kg=kg,
        Lz=Lz,
        extra=extra,
    )
    np.save(out[:-4] + "_gid.npy", gid)
    with open(out[:-4] + ".kf", "w") as fo:
        fo.write("%.17g %.17g %.17g\n" % (meta["K0_GPa_sqrt_h"], a0, y0))
        fo.write(" ".join(("%.17g" % v for v in kfnum)) + "\n")
    json.dump(
        {
            "mode": mode,
            "sigma": sigma,
            "angle": ang,
            "Lz": Lz,
            "W": W,
            "H": H,
            "a0": a0,
            "xg": xg,
            "f": float(f),
            "y0": float(y0),
            "N": int(m.N),
            "r": float(r),
            "l_h": lh,
            "wopt": wopt,
            "interp": interp,
            "hmls": float(hmls),
            "shift": list(shv) if mode in ("shift", "shiftpar") else None,
            "dpar": float(dpar) if mode == "shiftpar" else None,
            "gbang": gbang,
        },
        open(out[:-4] + ".json", "w"),
        indent=1,
    )


def case_bxn(
    out,
    mode="twist",
    sigma=5,
    f=0.0,
    Lz=None,
    W=56.0,
    H=36.0,
    a0=12.0,
    xg=16.0,
    disc="PXK_s1_f0",
    r=0.0,
    shift=None,
    dpar=2.0,
):
    W, H, a0, xg = (float(W), float(H), float(a0), float(xg))
    "the same bicrystal crossing test with the J04 / PX grain-boundary construction (gb_core.GBModel: overlap merge,\n    bonds by distance) and the PX per-boundary calibration (gb_poly.set_poly_gb_law: GB bond separation energy so that\n    the boundary cleaves at (1 - f)(2gamma_A + 2gamma_B)/2), K-field boundary as case_bx."
    from .grain_geometry import R_A, rotx, rotz, grain_frame, GBModel, law_from_spec
    from .polycrystal_geometry import set_poly_gb_law
    from .anisotropic_loading import GS

    sigma = int(sigma)
    m_, n_ = TRIPLES[sigma]
    ang = np.degrees(np.arctan2(m_, n_)) if mode not in ("shift", "shiftpar") else 0.0
    Lz = float(Lz) if Lz else float(sigma) if mode == "twist" else 5.0
    Q = rotx(ang) if mode == "twist" else rotz(ang)
    frames = [R_A, grain_frame(Q)] if mode not in ("shift", "shiftpar") else [R_A, R_A]
    assign = lambda P: (P[:, 0] >= xg).astype(int)
    if mode == "shiftpar":
        from .grain_geometry import crack_y0

        ypl = crack_y0(R_A, Lz) + float(dpar)
        assign = lambda P: (P[:, 1] >= ypl).astype(int)
    shv = tuple((float(v) for v in shift.split(","))) if shift else SHIFT_BX
    g = GBModel(
        frames,
        assign,
        kind="strip",
        W=W,
        H=H,
        a0=a0,
        Lz=Lz,
        shifts=[None, shv] if mode in ("shift", "shiftpar") else None,
    )
    L = law_from_spec(
        "uber" if not float(r) > 0 else "aniso:%g" % float(r), g.p, GS["300K"]
    )
    seeds = (
        np.array([[xg - 8.0, g.y0], [xg + 8.0, g.y0]])
        if mode != "shiftpar"
        else np.array([[W / 2, ypl - 8.0], [W / 2, ypl + 8.0]])
    )
    info = set_poly_gb_law(g, L, float(f), seeds)
    gpair = (g.l / L.l[g.sh]) ** 2
    meta = json.load(open_geometry(disc + ".json"))
    _, kfnum = read_disc_particles(disc)
    X, y0 = (g.md.X, g.y0)
    m = naive_lpm(g.md, g.p)
    pair = m.uid
    dmin = np.minimum.reduce(
        [X[:, 0], W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
    )
    dbc = np.minimum.reduce([W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]])
    fixed = np.repeat(dbc < 1.5, 3)
    protp = dmin < 5.5
    prb = protp[m.i] | protp[m.j]
    cutn = g.cross & (g.xm < a0)
    flag = prb.astype(np.uint8) << 1 | cutn[pair].astype(np.uint8) << 2
    gg = {}
    for b in np.nonzero(g.gbb & (np.abs(gpair - 1) > 1e-12))[0]:
        gg[g.Ip[b], g.Jp[b]] = gg[g.Jp[b], g.Ip[b]] = gpair[b]
    gm = np.array([gg.get((a, b), 1.0) for a, b in zip(m.i, m.j)])
    coh = (flag & 2 == 0) & (flag & 4 == 0)
    print(
        "bxn %s Sigma %d angle %.3f Lz %g: %d particles (%d merged), %d GB pairs, g %.3f-%.3f, info %s"
        % (
            mode,
            sigma,
            ang,
            Lz,
            len(X),
            g.nmerge,
            int(g.gbb.sum()),
            gm[gm != 1].min() if (gm != 1).any() else 1,
            gm.max(),
            {
                k: round(v["s2"], 3) if isinstance(v, dict) else v
                for k, v in info.items()
            },
        )
    )
    lh = l_h_for(r, meta)
    extra = {"hdr": [lh[0], lh[1], y0, H, W, a0], "coh": coh, "gmul": gm}
    write(
        out,
        X,
        fixed,
        np.zeros(3 * len(X)),
        m,
        m.kn0,
        g.p["T"],
        flag,
        np.ones(len(m.i)),
        pair,
        s_c=0.0746,
        tie=1e-06,
        kg=1e-07 * g.p["Kn"][0],
        Lz=Lz,
        extra=extra,
    )
    with open(out[:-4] + ".kf", "w") as fo:
        fo.write("%.17g %.17g %.17g\n" % (meta["K0_GPa_sqrt_h"], a0, y0))
        fo.write(" ".join(("%.17g" % v for v in kfnum)) + "\n")
    json.dump(
        {
            "mode": mode,
            "sigma": sigma,
            "angle": ang,
            "Lz": Lz,
            "W": W,
            "H": H,
            "a0": a0,
            "xg": xg,
            "f": float(f),
            "y0": float(y0),
            "N": int(len(X)),
            "model": "J04",
            "r": float(r),
            "l_h": lh,
            "shift": list(shv) if mode in ("shift", "shiftpar") else None,
            "dpar": float(dpar) if mode == "shiftpar" else None,
        },
        open(out[:-4] + ".json", "w"),
        indent=1,
    )
