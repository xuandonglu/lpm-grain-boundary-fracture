"""Rotated BCC grain clouds, overlap merging and cohesive model geometry.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np
from scipy.spatial import cKDTree
from .bcc_lattice import BCCModel, params, unit, BV, BSH, miller_frame
from .lattice_sites import sites
from .anisotropic_loading import NEP_C, GS, kfield_disp, lambda22, A_NEP
from .cleavage_calibration import calib_bond
from .cohesive_laws import UCDisc, Law, shape_g

R_A = miller_frame((1, 0, 0), (0, 1, 0))
S3_N = 1.2821


def rotz(deg):
    t = np.radians(deg)
    c, s = (np.cos(t), np.sin(t))
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def rotx(deg):
    t = np.radians(deg)
    c, s = (np.cos(t), np.sin(t))
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def roty(deg):
    t = np.radians(deg)
    c, s = (np.cos(t), np.sin(t))
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def grain_frame(Q):
    """grain frame for a specimen-frame rotation Q applied to the grain-1 lattice: X_spec = Q R_A pts."""
    return Q @ R_A


def law_from_spec(spec, p, gs):
    if spec == "uber":
        kinds, pars = (["uber", "uber"], [0.0, 0.0])
    elif spec.startswith("aniso"):
        r = float(spec.split(":")[1])
        gs = {"100": gs["100"], "110": r * gs["100"]}
        kinds, pars = (["uber", "uber"], [0.0, 0.0])
    elif spec.startswith("sexp"):
        n = float(spec.split(":")[1]) if ":" in spec else S3_N
        kinds, pars = (["sexp", "sexp"], [n, n])
    else:
        raise ValueError(spec)
    E, _ = calib_bond(p, gs)
    tmp = Law(p, kinds, pars, [1.0, 1.0])
    return Law(p, kinds, pars, np.sqrt(E / (tmp.k0 * tmp.c)))


def gamma_bb(m_cryst, Es):
    """broken-bond (full separation) energy 2gamma of a plane with crystal normal m for per-shell bond energies Es
    (GPa*a): 2gamma = sum_{b.m>0} rho_b E_b, rho_b = 2 (b.m) sites/a^3."""
    m = unit(m_cryst)
    bm = BV @ m
    s = bm > 1e-12
    return float(np.sum(2 * bm[s] * np.asarray(Es)[BSH[s]]))


def _periodic_tree(X, Lz):
    Xs = X - X.min(0)
    Xs[:, 2] = np.mod(X[:, 2], Lz)
    Xs[Xs[:, 2] >= Lz, 2] -= Lz
    Xs[Xs[:, 2] < 0, 2] = 0.0
    box = np.array([Xs[:, 0].max() * 3 + 10, Xs[:, 1].max() * 3 + 10, Lz])
    return cKDTree(Xs, boxsize=box)


def merge_overlaps(X, gid, rmerge, Lz, maxit=20):
    """Chen-Jiao-Liu 2015 replacing procedure: pairs closer than rmerge -> one particle at the midpoint (gid -1)."""
    X = X.copy()
    gid = gid.copy()
    nmerge = 0
    for it in range(maxit):
        tree = _periodic_tree(X, Lz)
        pr = tree.query_pairs(rmerge, output_type="ndarray")
        if len(pr) == 0:
            break
        used = np.zeros(len(X), bool)
        keep = np.ones(len(X), bool)
        for i, j in pr:
            if used[i] or used[j]:
                continue
            d = X[j] - X[i]
            d[2] -= Lz * np.round(d[2] / Lz)
            X[i] = X[i] + 0.5 * d
            X[i, 2] = np.mod(X[i, 2], Lz)
            gid[i] = -1
            keep[j] = False
            used[i] = used[j] = True
            nmerge += 1
        X = X[keep]
        gid = gid[keep]
    return (X, gid, nmerge)


def build_cloud(frames, assign, lo, hi, Lz, rmerge=0.52, shifts=None):
    """frames: list of Rm per grain; assign(X) -> grain index array; shifts: specimen-frame rigid translation of
    each grain lattice (GB-structure variation).  Returns X, gid, nmerge."""
    Xs, G = ([], [])
    for g, Rm in enumerate(frames):
        sh = (
            None
            if shifts is None or shifts[g] is None
            else Rm.T @ np.asarray(shifts[g], float)
        )
        Xg = sites(Rm, (lo[0], lo[1], 0.0), (hi[0], hi[1], Lz), shift=sh)
        k = assign(Xg) == g
        Xs.append(Xg[k])
        G.append(np.full(k.sum(), g))
    X = np.vstack(Xs)
    gid = np.concatenate(G)
    return merge_overlaps(X, gid, rmerge, Lz)


def crack_y0(Rm1, Lz):
    X = sites(Rm1, (-3, -3, 0), (3, 3, Lz))
    ys = np.unique(np.round(X[:, 1], 6))
    k = np.searchsorted(ys, 0.0)
    return 0.5 * (ys[k - 1] + ys[k])


class GBModel(UCDisc):

    def __init__(
        self,
        frames,
        assign,
        kind="disc",
        Rd=20.0,
        x_gb=-0.25,
        x_tip=-3.0,
        W=60.0,
        H=36.0,
        a0=10.0,
        Lz=3.0,
        C=NEP_C,
        shell=1.5,
        prot=4.0,
        rmerge=0.52,
        grip=1.2,
        shifts=None,
    ):
        p = params(*C)
        self.p, self.Cel, self.Lz, self.frames, self.kind = (p, C, Lz, frames, kind)
        self.R = frames[0]
        y0 = crack_y0(frames[0], Lz)
        self.y0 = y0
        self.x_gb = x_gb
        if kind == "disc":
            lo = (x_gb - Rd - 2, y0 - Rd - 2)
            hi = (x_gb + Rd + 2, y0 + Rd + 2)
        else:
            lo = (0.0, y0 - H / 2)
            hi = (W, y0 + H / 2)
        X, gid, nm = build_cloud(frames, assign, lo, hi, Lz, rmerge, shifts)
        if kind == "disc":
            r = np.hypot(X[:, 0] - x_gb, X[:, 1] - y0)
            k = r <= Rd + 1e-09
            X, gid = (X[k], gid[k])
        self.gid = gid
        self.nmerge = nm
        self.md = md = BCCModel(X, frames[0], p, Lz=Lz)
        self.N = N = md.N
        P = md.npair
        self.P = P
        I, J = (md.I[:P], md.J[:P])
        self.Ip, self.Jp = (I, J)
        self.sh = md.sh[:P]
        self.L0 = md.L0[:P]
        self.n = md.n[:P]
        Xi, Xj = (X[I], X[J])
        self.xm = 0.5 * (Xi[:, 0] + Xj[:, 0])
        self.ym = 0.5 * (Xi[:, 1] + Xj[:, 1])
        dz = md.d[:P, 2]
        self.zm = np.mod(Xi[:, 2] - 0.5 * dz, Lz)
        self.gbb = (gid[I] != gid[J]) | (gid[I] < 0) | (gid[J] < 0)
        self.bgrain = np.where(self.gbb, -1, gid[I])
        self.cross = (Xi[:, 1] - y0) * (Xj[:, 1] - y0) < 0
        if kind == "disc":
            self.Rd = Rd
            self.xc = x_tip
            self.cut = self.cross & (self.xm < x_tip) & (self.bgrain == 0)
            rr = np.hypot(X[:, 0] - x_gb, X[:, 1] - y0)
            self.rr = rr
            bc = rr > Rd - shell
            self.protp = rr > Rd - shell - prot
            uK = kfield_disp(
                C, frames[0], X[:, 0] - x_gb, X[:, 1] - y0, (0.0, 1.0, 0.0)
            )
        else:
            self.Rd = max(W, H)
            self.xc = a0
            self.W, self.H = (W, H)
            self.cut = self.cross & (self.xm < a0)
            top = X[:, 1] > y0 + H / 2 - grip
            bot = X[:, 1] < y0 - H / 2 + grip
            bc = top | bot
            self.protp = (X[:, 1] > y0 + H / 2 - grip - prot) | (
                X[:, 1] < y0 - H / 2 + grip + prot
            )
            uK = np.zeros((N, 3))
            uK[top, 1] = 0.5
            uK[bot, 1] = -0.5
        self.fixed = np.sort(
            np.concatenate([3 * np.where(bc)[0] + c for c in range(3)])
        )
        self.uK = uK.ravel()
        self.free = np.ones(3 * N, bool)
        self.free[self.fixed] = False
        self.protb = self.protp[I] | self.protp[J]
        self.Bp = md.B[:P].tocsr()
        self.Bd = md.B.tocsr()
        self.S = md.S.tocsr()
        self.rowI = 2 * I + self.sh
        self.rowJ = 2 * J + self.sh
        self.k0 = 2 * p["Kn"][self.sh].astype(float)
        self.coh = np.zeros(P, bool)
        self.E = np.zeros(P)
        self.l = np.ones(P)
        self.alive = (~self.cut).astype(float)
        self.T = p["T"]
        self.kg = 1e-07 * p["Kn"][0]
        self.u = np.zeros(3 * N)
        self._kfit_cache = {}
        self.Lam22 = lambda22(C, frames[0])[1, 1]
        self.kind_s = ["uber", "uber"]
        self.par_s = [0.0, 0.0]
        self._g = [shape_g("uber", 0)] * 2
        self.brittle = False

    def gb_stats(self, ywin=8.0):
        """bonds across the GB plane x = x_gb (disc) in |y - y0| < ywin: count, shells, length histogram."""
        X = self.md.X
        I, J = (self.Ip, self.Jp)
        crossp = (X[I, 0] - self.x_gb) * (X[J, 0] - self.x_gb) < 0
        win = crossp & (self.ym - self.y0 >= -ywin) & (self.ym - self.y0 < ywin)
        A = 2 * ywin * self.Lz
        g = self.gbb
        out = dict(
            n_merge=int(self.nmerge),
            n_gb_particles=int((self.gid < 0).sum()),
            n_gb_bonds_total=int(g.sum()),
            cross_per_area=float(win.sum() / A),
            cross_gb_per_area=float((win & g).sum() / A),
            cross_sh1_per_area=float((win & (self.sh == 0)).sum() / A),
            cross_sh2_per_area=float((win & (self.sh == 1)).sum() / A),
            L0_hist=np.histogram(
                self.L0[win & g], bins=[0.5, 0.6, 0.7, 0.8, 0.875, 0.95, 1.01]
            )[0].tolist(),
            L0_min=float(self.L0[g].min()) if g.any() else None,
        )
        return (out, win, A)

    def set_gb_law(self, law, f=0.0, gamma12=None, ywin=8.0):
        """bulk law on all bonds; GB bonds: same shape, l scaled so that the GB-plane cleavage energy equals
        (1 - f)(gamma_1 + gamma_2).  gamma12 = (2gamma_1 + 2gamma_2)/2 target average in GPa*a (2gamma units).
        """
        self.set_law(law)
        st, win, A = self.gb_stats(ywin)
        g = self.gbb & self.coh
        Ecross = (
            float(np.sum((self.k0 * self.l**2 * law.c[self.sh])[win & self.coh])) / A
        )
        Eint = (
            float(
                np.sum(
                    (self.k0 * self.l**2 * law.c[self.sh])[win & self.coh & ~self.gbb]
                )
            )
            / A
        )
        Egb = float(np.sum((self.k0 * self.l**2 * law.c[self.sh])[win & g])) / A
        target = (1 - f) * gamma12 if gamma12 is not None else (1 - f) * Ecross
        s2 = max((target - Eint) / Egb, 1e-06) if Egb > 0 else 1.0
        s = np.sqrt(s2)
        self.l[g] *= s
        self.E[g] *= s2
        self.gb_scale = dict(
            f=f,
            s=float(s),
            s2=float(s2),
            twoG_raw=Ecross,
            twoG_target=float(target),
            twoG_intra=Eint,
            twoG_gbbonds=Egb,
            stats=st,
        )
        return self.gb_scale


def KIG_A(gs="300K", C=NEP_C):
    """K_IG of the grain-1 (100)[010] crack with the NEP gamma (GPa sqrt(a))."""
    G2 = 2 * GS[gs]["100"] / (1000000000.0 * A_NEP)
    return float(np.sqrt(G2 / lambda22(C, R_A)[1, 1]))
