"""Bicrystal export.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

from ..paths import open_geometry
import json
import os
import numpy as np
from . import model_export as ge


def frames_of(psi, sym=False):
    from .grain_geometry import R_A, rotz, grain_frame

    if sym:
        return [grain_frame(rotz(+psi / 2)), grain_frame(rotz(-psi / 2))]
    return [R_A, grain_frame(rotz(psi))] if psi else [R_A, R_A]


def export(
    kind,
    out,
    psi=0.0,
    f=0.0,
    dpar=0.0,
    shift=None,
    W=48.0,
    H=28.0,
    a0=12.0,
    xg=16.0,
    Lz=5.0,
    r=0.0,
    disc=None,
    hmls=0.9,
    sym=0,
):
    from .grain_geometry import crack_y0, R_A

    psi, f, dpar, W, H, a0, xg, Lz, r = (
        float(v) for v in (psi, f, dpar, W, H, a0, xg, Lz, r)
    )
    disc = disc or (
        {"W": "WK_s10", "Mo": "MoK_s10"}.get(os.environ.get("MAT")) or "PXK_s1_f0"
    )
    jonly = os.environ.get("JONLY") == "1"
    strip = os.environ.get("LOAD") == "strip"
    if strip:
        a0 = 0.0
    ustrip = float(os.environ.get("USTRIP", "3"))
    frames = frames_of(psi, sym=bool(int(sym)))
    shv = tuple((float(v) for v in shift.split(","))) if shift else None
    shifts = [None, shv] if shv else None
    y0 = crack_y0(R_A, Lz)
    ypl = y0 + dpar
    assign = lambda P: (P[:, 1] >= ypl).astype(int)
    meta = json.load(open_geometry(disc + ".json"))
    _, kfnum = ge.read_disc_particles(disc)
    p = ge.params(ge.NEP["C11"], ge.NEP["C12"], ge.NEP["C44"], a=1.0, mapping="paper")
    kg = 1e-07 * p["Kn"][0]
    lo, hi = ((0.0, y0 - H / 2), (W, y0 + H / 2))
    if kind == "ghost":
        X, gid = ge.build_grains(frames, assign, lo, hi, Lz, shifts=shifts)
        X = np.round(X, 10)
        X[:, 2] = np.mod(X[:, 2], Lz)
        X[X[:, 2] >= Lz - 1e-09, 2] = 0.0
        m = ge.GhostLPM(
            X,
            gid,
            frames,
            assign,
            ge.NEP,
            Lz,
            lo,
            hi,
            crack_aware=False,
            interp="mls",
            hmls=float(hmls),
        )
        wopt = os.environ.get("GH_WOPT")
        if wopt:
            from .weight_optimization import (
                optimise_weights_gen,
                set_weights,
                exact_bicrystal_n,
            )
            from .elasticity import cubic_C, rotate_C

            C0 = cubic_C(ge.NEP["C11"], ge.NEP["C12"], ge.NEP["C44"])
            CA, CB = (rotate_C(C0, frames[0]), rotate_C(C0, frames[1]))
            x0 = np.array([0.0, ypl, 0.0])
            flds = []
            for E in (
                np.diag([1.0, 0, 0]),
                np.diag([0, 1.0, 0]),
                np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
            ):
                HA, HB = exact_bicrystal_n(E, CA, CB, [0.0, 1.0, 0.0])
                flds.append(
                    lambda P, HA=HA, HB=HB: np.where(
                        (P[:, 1] < ypl)[:, None], (P - x0) @ HA.T, (P - x0) @ HB.T
                    )
                )
            dbc0 = np.minimum.reduce(
                [W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
            )
            wd, winfo = optimise_weights_gen(
                m, (dbc0 >= 2.6) & (X[:, 0] > 2.6), flds, mu=float(wopt)
            )
            set_weights(m, wd)
        pair = m.pair
        fixed, protp, ubc = bcs(X, y0, H, W, strip, ustrip)
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
        if jonly:
            gp = np.zeros(m.nb, bool)
            gp[pair[m.ghost]] = True
            prp |= ~gp
        flag = (
            m.ghost.astype(np.uint8)
            | prp[pair].astype(np.uint8) << 1
            | cutp[pair].astype(np.uint8) << 2
        )
        gm = np.where(m.ghost, 1.0 - f, 1.0)
        kn, T, nb = (m.Kn, m.T, int(m.ghost.sum()))
        np.save(out[:-4] + "_gid.npy", gid)
        print(
            "ghost model: %d particles, %d bonds, %d ghost, %d MLS fallbacks"
            % (m.N, m.nb, nb, m.nfallback)
        )
    else:
        from . import polycrystal_geometry as gb_poly
        from .grain_geometry import GBModel, law_from_spec
        from .anisotropic_loading import GS

        g = GBModel(frames, assign, kind="strip", W=W, H=H, a0=a0, Lz=Lz, shifts=shifts)
        L = law_from_spec("uber" if not r > 0 else "aniso:%g" % r, g.p, GS["300K"])
        seeds = np.array([[W / 2, ypl - 8.0], [W / 2, ypl + 8.0]])
        if kind == "emc":
            from .energy_matching import emc_law

            info = emc_law(g, L, f, seeds)
        elif kind == "px":
            info = gb_poly.set_poly_gb_law(g, L, f, seeds)
        elif kind == "plain":
            g.set_law(L)
            g.l[g.gbb] *= np.sqrt(1.0 - f)
            g.E[g.gbb] *= 1.0 - f
            info = {"plain": True}
        else:
            raise SystemExit("kind must be ghost, emc, plain or px")
        gpair = (g.l / L.l[g.sh]) ** 2
        X = g.md.X
        m = ge.naive_lpm(g.md, g.p)
        pair = m.uid
        fixed, protp, ubc = bcs(X, y0, H, W, strip, ustrip)
        prb = protp[m.i] | protp[m.j]
        if jonly:
            prb |= ~g.gbb[pair]
        cutn = g.cross & (g.xm < a0)
        flag = prb.astype(np.uint8) << 1 | cutn[pair].astype(np.uint8) << 2
        gg = {}
        for b in np.nonzero(g.gbb & (np.abs(gpair - 1) > 1e-12))[0]:
            gg[g.Ip[b], g.Jp[b]] = gg[g.Jp[b], g.Ip[b]] = gpair[b]
        gm = np.array([gg.get((a, b), 1.0) for a, b in zip(m.i, m.j)])
        kn, T, nb = (m.kn0, g.p["T"], int(g.gbb.sum()))
        print(
            "merged (%s): %d particles (%d merged), %d boundary pairs, g %.3f-%.3f, info %s"
            % (
                kind,
                len(X),
                g.nmerge,
                nb,
                gm.min(),
                gm.max(),
                {
                    k: round(v["s2"], 3) if isinstance(v, dict) and "s2" in v else v
                    for k, v in info.items()
                },
            )
        )
    coh = (flag & 2 == 0) & (flag & 4 == 0)
    lh = ge.l_h_for(r, meta)
    extra = {"hdr": [lh[0], lh[1], y0, H, W, a0], "coh": coh, "gmul": gm}
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
        s_c=0.0746,
        tie=1e-06,
        kg=kg,
        Lz=Lz,
        extra=extra,
    )
    if strip:
        if os.path.exists(out[:-4] + ".kf"):
            os.remove(out[:-4] + ".kf")
    else:
        with open(out[:-4] + ".kf", "w") as fo:
            fo.write("%.17g %.17g %.17g\n" % (meta["K0_GPa_sqrt_h"], a0, y0))
            fo.write(" ".join(("%.17g" % v for v in kfnum)) + "\n")
    json.dump(
        {
            "mode": "par",
            "kind": kind,
            "psi": psi,
            "f": f,
            "dpar": dpar,
            "shift": list(shv) if shv else None,
            "Lz": Lz,
            "W": W,
            "H": H,
            "a0": a0,
            "xg": xg,
            "y0": float(y0),
            "N": int(len(X)),
            "r": r,
            "l_h": lh,
            "jonly": jonly,
            "material": os.environ.get("MAT", "NEP-Fe"),
            "disc": disc,
            "wopt": os.environ.get("GH_WOPT"),
            "sigma": 0,
            "angle": psi,
            "sym": int(sym),
            "load": "strip" if strip else "K",
            "ustrip": ustrip if strip else None,
        },
        open(out[:-4] + ".json", "w"),
        indent=1,
    )


def bcs(X, y0, H, W, strip, ustrip):
    """fixed dofs, protected particles and the boundary displacement per unit load: K field (as case_bx: outer 1.5 h prescribed
    by the driver, 5.5 h protected) or clamped strip (as case_px: grips 1.2 h, +-ustrip/2, 4 h protection)
    """
    if strip:
        grip, prot = (1.2, 4.0)
        top, bot = (X[:, 1] > y0 + H / 2 - grip, X[:, 1] < y0 - H / 2 + grip)
        protp = (X[:, 1] > y0 + H / 2 - grip - prot) | (
            X[:, 1] < y0 - H / 2 + grip + prot
        )
        ubc = np.zeros((len(X), 3))
        ubc[top, 1], ubc[bot, 1] = (0.5 * ustrip, -0.5 * ustrip)
        return (np.repeat(top | bot, 3), protp, ubc.ravel())
    dmin = np.minimum.reduce(
        [X[:, 0], W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]]
    )
    dbc = np.minimum.reduce([W - X[:, 0], X[:, 1] - (y0 - H / 2), y0 + H / 2 - X[:, 1]])
    return (np.repeat(dbc < 1.5, 3), dmin < 5.5, np.zeros(3 * len(X)))
