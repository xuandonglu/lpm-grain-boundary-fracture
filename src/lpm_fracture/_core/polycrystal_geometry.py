"""Grain-boundary energy calibration helper for model construction.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import os

os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
import numpy as np
from .grain_geometry import gamma_bb


def set_poly_gb_law(m, L, f, seeds, h=0.1):
    """per-GB calibration: for every pair of neighbouring grains (g, h) the GB bonds connecting them get
    l *= s_gh with s_gh^2 = (1 - f) * (2gamma_g(n) + 2gamma_h(n))/2 / (raw GB-plane cleavage energy per area),
    n = boundary normal (Voronoi bisector); boundary length from a rasterised Voronoi map.
    """
    m.set_law(L)
    X = m.md.X
    lab = np.argmin(((X[:, None, :2] - seeds[None, :, :]) ** 2).sum(-1), axis=1)
    li, lj = (lab[m.Ip], lab[m.Jp])
    g = m.gbb & m.coh
    xs = np.arange(0, m.W, h) + h / 2
    ys = m.y0 - m.H / 2 + np.arange(0, m.H, h) + h / 2
    GX, GY = np.meshgrid(xs, ys, indexing="ij")
    Lg = np.argmin(
        (GX[..., None] - seeds[:, 0]) ** 2 + (GY[..., None] - seeds[:, 1]) ** 2, axis=-1
    )
    Eb = m.k0 * m.l**2 * L.c[m.sh]
    info = {}
    s2_all = []
    for a in range(len(seeds)):
        for b in range(a + 1, len(seeds)):
            nvec = seeds[b] - seeds[a]
            nvec = nvec / np.linalg.norm(nvec)
            cx = np.sum(
                (Lg[:-1, :] == a) & (Lg[1:, :] == b)
                | (Lg[:-1, :] == b) & (Lg[1:, :] == a)
            )
            cy = np.sum(
                (Lg[:, :-1] == a) & (Lg[:, 1:] == b)
                | (Lg[:, :-1] == b) & (Lg[:, 1:] == a)
            )
            if cx + cy == 0:
                continue
            Lb = (
                cx * h / abs(nvec[0])
                if abs(nvec[0]) > abs(nvec[1])
                else cy * h / abs(nvec[1])
            )
            sel = g & ((li == a) & (lj == b) | (li == b) & (lj == a))
            if sel.sum() == 0 or Lb < 1.0:
                continue
            raw = Eb[sel].sum() / (Lb * m.Lz)
            n3 = np.array([nvec[0], nvec[1], 0.0])
            tg = 0.5 * (
                gamma_bb(m.frames[a].T @ n3, L.E) + gamma_bb(m.frames[b].T @ n3, L.E)
            )
            s2 = (1 - f) * min(tg / raw, 4.0)
            m.l[sel] *= np.sqrt(s2)
            m.E[sel] *= s2
            s2_all.append(s2)
            info[f"{a}-{b}"] = dict(
                length=float(Lb),
                raw2G=float(raw),
                target2G=float((1 - f) * tg),
                s2=float(s2),
                nbonds=int(sel.sum()),
            )
    other = g & (li == lj)
    if other.any() and s2_all:
        m.l[other] *= np.sqrt(np.mean(s2_all))
        m.E[other] *= np.mean(s2_all)
    info["n_samelabel_gb_bonds"] = int(other.sum())
    m.gb_scale = info
    return info
