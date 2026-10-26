"""Convert physical cleavage energies to bond energies and opening scales.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np
from .bcc_lattice import BV, BSH, unit
from .anisotropic_loading import A_NEP


def crossing(m, a=1.0):
    """bond types crossing a plane with normal m (crystal frame): density per unit area rho_b (undirected, counted
    once), cos c_b = |b.m|/L0, shell; plus layer structure for exact rigid separation.
    """
    m = unit(m)
    b = BV * a
    L0 = np.linalg.norm(b, axis=1)
    bm = b @ m
    sel = bm > 1e-12
    rho = 2 / a**3 * bm[sel]
    return dict(
        rho=rho, c=bm[sel] / L0[sel], sh=BSH[sel], L0=L0[sel], bm=bm[sel], b=b[sel]
    )


def rigid_sep_curve(m, law, p, deltas, a=1.0):
    """exact rigid separation energy per area E(delta) and traction for a given bond-law function
    law(sh, cos) -> (E, l) or None (harmonic).  Multibody included via layer classes."""
    m = unit(m)
    b = BV * a
    L0 = np.linalg.norm(b, axis=1)
    bm = b @ m
    pts = np.array(
        [(i, j, k) for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)],
        float,
    )
    pts = np.vstack([pts, pts + 0.5]) * a
    hs = np.unique(np.round(pts @ m, 9))
    dsp = np.min(np.diff(hs))
    rho_layer = 2 / a**3 * dsp
    K = int(np.ceil(np.abs(bm).max() / dsp)) + 1
    z_low = -dsp / 2 - dsp * np.arange(K)
    T = p["T"]
    out = []
    for d in deltas:
        Epair = 0.0
        Emb = 0.0
        for z in z_low:
            cr = z + bm > 0
            for sgn in (1, -1):
                Ds = np.zeros(2)
                for q in np.where(cr)[0]:
                    c = bm[q] / L0[q]
                    x = d * c
                    E, l = law(BSH[q], c)
                    if E is None:
                        ph = p["Kn"][BSH[q]] * x**2
                        ps = x
                    else:
                        ph = E * (1 - (1 + x / l) * np.exp(-x / l))
                        ps = x * np.exp(-x / l)
                    if sgn == 1:
                        Epair += rho_layer * ph
                    Ds[BSH[q]] += ps
                Emb += rho_layer * 0.5 * T * np.sum(Ds**2)
        out.append(Epair + Emb)
    E = np.array(out)
    return (E, np.gradient(E, deltas))


def calib_bond(p, gs, a_h=A_NEP):
    """bond-UBER (elastic-consistent) per-shell energies E_s solving 2gamma{100}, 2gamma{110} (exact, linear).
    gs in J/m^2; particle spacing a_h [m] (h).  Returns E (2,), l (2,)."""
    Gm = lambda g: 2 * g / (1000000000.0 * a_h)
    A = np.zeros((2, 2))
    rhs = np.zeros(2)
    for r, (f, m) in enumerate((("100", (1, 0, 0)), ("110", (1, 1, 0)))):
        cr = crossing(m)
        for s in (0, 1):
            A[r, s] = cr["rho"][cr["sh"] == s].sum()
        rhs[r] = Gm(gs[f])
    E = np.linalg.solve(A, rhs)
    l = np.sqrt(E / (2 * p["Kn"]))
    return (E, l)
