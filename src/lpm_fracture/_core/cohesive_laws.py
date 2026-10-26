"""Cohesive energy shapes and per-shell parameters used in model preparation.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np
from scipy.special import gamma as Gfun
from .bcc_lattice import BV, BSH


def shape_g(kind, par):
    """returns g(t) (vectorised), g'(t), and c = int_0^inf t g(t) dt (energy factor E = k0 l^2 c)."""
    if kind == "uber":
        g = lambda t: np.exp(-t)
        dg = lambda t: -np.exp(-t)
        c = 1.0
    elif kind == "sexp":
        n = float(par)
        g = lambda t: np.exp(-(t**n) / n)
        dg = lambda t: -(t ** (n - 1)) * np.exp(-(t**n) / n)
        c = n ** (2 / n - 1) * Gfun(2 / n)
    elif kind == "rfs":
        f = float(par)
        g = lambda t: (1 - 3 * f * t + f * t**2) * np.exp(-t)
        dg = lambda t: (-3 * f + 2 * f * t - (1 - 3 * f * t + f * t**2)) * np.exp(-t)
        c = 1.0 + f * (6 - 6)
    else:
        raise ValueError(kind)
    return (g, dg, c)


def phi_of(kind, par, t):
    """phi/(k0 l^2) = int_0^t s g(s) ds (vectorised via closed forms where available)."""
    t = np.asarray(t, float)
    if kind == "uber":
        return 1 - (1 + t) * np.exp(-t)
    if kind == "rfs":
        f = float(par)
        return 1 - (1 + t + f * t**3) * np.exp(-t)
    if kind == "sexp":
        from scipy.special import gammainc

        n = float(par)
        return n ** (2 / n - 1) * Gfun(2 / n) * gammainc(2 / n, t**n / n)
    raise ValueError(kind)


class Law:
    """per-shell cohesive bond law: kinds[s], pars[s], l[s]; k0[s] = 2 Kn_s (exact small-strain stiffness)."""

    def __init__(self, p, kinds, pars, ls):
        self.kinds = list(kinds)
        self.pars = [float(x) for x in pars]
        self.l = np.asarray(ls, float)
        self.k0 = 2 * np.asarray(p["Kn"], float)
        self.c = np.array([shape_g(k, q)[2] for k, q in zip(self.kinds, self.pars)])
        self.E = self.k0 * self.l**2 * self.c

    def phi(self, s, x):
        t = np.maximum(x, 0) / self.l[s]
        return np.where(
            x > 0,
            self.k0[s] * self.l[s] ** 2 * phi_of(self.kinds[s], self.pars[s], t),
            0.5 * self.k0[s] * x**2,
        )

    def psi(self, s, x):
        g, _, _ = shape_g(self.kinds[s], self.pars[s])
        t = np.maximum(x, 0) / self.l[s]
        return np.where(x > 0, x * g(t), x)

    def as_dict(self):
        return dict(
            kinds=self.kinds,
            pars=self.pars,
            l=self.l.tolist(),
            E=self.E.tolist(),
            k0=self.k0.tolist(),
        )


def rigid_sep(m, law, p, deltas):
    """exact rigid separation energy/area and traction (GPa) of plane normal m for a Law (multibody included)."""
    m = np.asarray(m, float)
    m = m / np.linalg.norm(m)
    b = BV * 1.0
    L0 = np.linalg.norm(b, axis=1)
    bm = b @ m
    pts = np.array(
        [(i, j, k) for i in range(-3, 4) for j in range(-3, 4) for k in range(-3, 4)],
        float,
    )
    pts = np.vstack([pts, pts + 0.5])
    hs = np.unique(np.round(pts @ m, 9))
    dsp = np.min(np.diff(hs))
    rho_layer = 2 * dsp
    K = int(np.ceil(np.abs(bm).max() / dsp)) + 1
    z_low = -dsp / 2 - dsp * np.arange(K)
    T = p["T"]
    deltas = np.asarray(deltas, float)
    Epair = np.zeros_like(deltas)
    Emb = np.zeros_like(deltas)
    for z in z_low:
        cr = np.where(z + bm > 0)[0]
        Ds = np.zeros((2, len(deltas)))
        for q in cr:
            c = bm[q] / L0[q]
            x = deltas * c
            s = BSH[q]
            Epair += rho_layer * law.phi(s, x)
            Ds[s] += law.psi(s, x)
        Emb += 2 * rho_layer * 0.5 * T * np.sum(Ds**2, axis=0)
    E = Epair * 1.0 + Emb
    return (E, np.gradient(E, deltas))


class UCDisc:

    def set_law(self, law, r_core=None):
        mask = ~self.protb & ~self.cut
        if r_core is not None:
            rm = np.hypot(self.xm - self.xc, self.ym - self.y0)
            mask &= rm < r_core
        self.coh[:] = False
        self.coh[mask] = True
        self.lawobj = law
        self.l = law.l[self.sh].copy()
        self.E = law.E[self.sh].copy()
        self.k0 = law.k0[self.sh].copy()
        self.kind_s = law.kinds
        self.par_s = law.pars
        self._g = [shape_g(k, q) for k, q in zip(law.kinds, law.pars)]
