"""Two-dimensional bond softening with visibility screening and reverse-half-bond pairing.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import math
import os
import time
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from scipy.sparse.csgraph import connected_components
from scipy.spatial import cKDTree
from .planar_geometry import (
    hex_points,
    assign,
    merge_overlaps,
    mls_weights,
    tri_weights,
)
from .hexagonal_stiffness import params_from_C

SQ3 = math.sqrt(3.0)
CAL = np.array([[108.24, 62.16, 0], [62.16, 108.24, 0], [0, 0, 28.41]])


def mls_weights_q(Y, h=1.2):
    """quadratic-exact MLS weights of the origin (basis 1, x, y, x^2, xy, y^2); falls back to linear MLS if singular"""
    d = np.linalg.norm(Y, axis=1)
    w = np.exp(-((d / h) ** 2))
    P = np.column_stack(
        [np.ones(len(Y)), Y, Y[:, 0] ** 2, Y[:, 0] * Y[:, 1], Y[:, 1] ** 2]
    )
    A = P.T @ (w[:, None] * P)
    if len(Y) < 6 or np.linalg.cond(A) > 10000000000.0:
        return mls_weights(Y, 1.0)
    e = np.zeros(6)
    e[0] = 1.0
    return w * (P @ np.linalg.solve(A, e))


class Frac2D:

    def __init__(
        self,
        W,
        seeds,
        theta,
        shift,
        C,
        kind,
        a_g=0.001,
        a_gb=0.001,
        nsup=12,
        hmls=1.0,
        grip=3.0,
        notch=0.0,
        interp=None,
    ):
        self.W, self.kind = (W, kind)
        interp = interp or os.environ.get("INTERP", "mls")
        self.interp = interp
        self._wfun = (
            (lambda Y: tri_weights(Y))
            if interp == "tri"
            else (
                (lambda Y: mls_weights_q(Y, 1.2))
                if interp == "quad"
                else lambda Y: mls_weights(Y, hmls)
            )
        )
        k, T = params_from_C(C)
        X, gid = ([], [])
        for g in range(len(seeds)):
            P = hex_points(theta[g], shift[g], (0, 0), (W, W))
            m = assign(P, seeds) == g
            X.append(P[m])
            gid.append(np.full(m.sum(), g))
        X, gid = (np.vstack(X), np.concatenate(gid))
        mflag = np.zeros(len(X), bool)
        if kind == "j04":
            X0 = X
            X, gid = merge_overlaps(X, gid, 0.6)
            mflag = cKDTree(X0).query(X)[0] > 1e-09
        self.X, self.gid, self.N = (X, gid, len(X))
        self.seeds = seeds
        I, J, NV, L0, KB, TB, GB, SUP, WS = ([], [], [], [], [], [], [], [], [])
        if kind == "j04":
            pr = cKDTree(X).query_pairs((1 + SQ3) / 2, output_type="ndarray")
            ii = np.r_[pr[:, 0], pr[:, 1]]
            jj = np.r_[pr[:, 1], pr[:, 0]]
            d = X[jj] - X[ii]
            L = np.linalg.norm(d, axis=1)
            nv = d / L[:, None]
            gb = (gid[ii] != gid[jj]) | mflag[ii] | mflag[jj]
            ang = np.arctan2(nv[:, 1], nv[:, 0]) - theta[gid[ii]]
            jt = np.round(np.mod(ang, math.pi) / (math.pi / 3)).astype(int) % 3
            if os.environ.get("GBMEAN", "1") == "1":
                kb = np.where(gb, k.mean(), k[jt])
                tb = np.where(gb, T.mean(), T[jt])
            else:
                kb, tb = (k[jt], T[jt])
            I, J, NV, L0, KB, TB, GB = (ii, jj, nv, L, kb, tb, gb)
            npair = len(pr)
            self.pair = np.r_[np.arange(npair), np.arange(npair)]
            self.rev = np.r_[np.arange(npair) + npair, np.arange(npair)]
            SUP = [None] * len(I)
            WS = [None] * len(I)
        else:
            trees = {
                g: (cKDTree(X[gid == g]), np.nonzero(gid == g)[0])
                for g in np.unique(gid)
            }
            self.cand = []
            for p in range(len(X)):
                g = gid[p]
                for Jd in range(6):
                    a = theta[g] + Jd * math.pi / 3
                    v = np.array([math.cos(a), math.sin(a)])
                    site = X[p] + v
                    inside = (
                        -1e-09 <= site[0] < W + 1e-09 and -1e-09 <= site[1] < W + 1e-09
                    )
                    h = assign(site[None], seeds)[0] if inside else g
                    if h == g:
                        dd, q = trees[g][0].query(site)
                        if dd > 1e-06:
                            continue
                        I.append(p)
                        J.append(trees[g][1][q])
                        SUP.append(None)
                        WS.append(None)
                        self.cand.append(None)
                        GB.append(False)
                    else:
                        dd, q = trees[h][0].query(site, k=min(nsup, len(trees[h][1])))
                        sup = trees[h][1][np.atleast_1d(q)]
                        I.append(p)
                        J.append(-1)
                        SUP.append(sup)
                        WS.append(self._wfun(X[sup] - site))
                        self.cand.append(sup)
                        GB.append(True)
                    NV.append(v)
                    L0.append(1.0)
                    KB.append(k[Jd % 3])
                    TB.append(T[Jd % 3])
            I, J, NV, L0, KB, TB, GB = map(np.asarray, (I, J, NV, L0, KB, TB, GB))
            key = {
                (int(a_), int(c_)): q
                for q, (a_, c_) in enumerate(zip(I.tolist(), J.tolist()))
                if c_ >= 0
            }
            rev = np.array(
                [
                    key.get((int(J[q]), int(I[q])), -1) if J[q] >= 0 else q
                    for q in range(len(I))
                ]
            )
            assert (rev >= 0).all()
            self.pair = np.minimum(np.arange(len(I)), rev)
            self.rev = rev
        self.I, self.J, self.n, self.L0, self.kb, self.tb, self.gbb = (
            I,
            J,
            NV,
            L0,
            KB,
            TB,
            GB,
        )
        self.sup, self.w = (SUP, WS)
        self.w_static = [None if w is None else w.copy() for w in WS]
        self.w_mls = [None if w is None else w.copy() for w in WS]
        self.relax_R = 0.0
        self.nb = len(I)
        self.dlc = np.where(GB, a_gb, a_g) * 0.5
        self.alive = np.ones(self.nb, bool)
        self.phi = np.ones(self.nb)
        self.efac = np.ones(self.nb)
        self.stage = np.zeros(self.nb, int)
        self.soft_m, self.soft_nt, self.soft_a = (1.0, 0, 0.5)
        self.bot = X[:, 1] < grip
        self.top = X[:, 1] > W - grip
        fixed = self.bot | self.top
        self.fixed = np.repeat(fixed, 2)
        self.free = ~self.fixed
        self.ug = np.zeros(2 * self.N)
        self.ug[2 * np.nonzero(self.top)[0] + 1] = 1.0
        self.rb = np.arange(self.nb)
        self.build()
        if notch > 0:
            self.cut_notch(notch)

    def cut_notch(self, notch):
        """cut the bonds crossing y = W/2 for x < notch (call AFTER any weight optimisation: the weights are fitted on the
        intact model, so that notch-face residuals are not 'corrected' with boundary weights)
        """
        W = self.W
        P1 = self.X[self.I]
        P2 = P1 + self.n * self.L0[:, None]
        cut = ((P1[:, 1] - W / 2) * (P2[:, 1] - W / 2) < 0) & (
            np.minimum(P1[:, 0], P2[:, 0]) < notch
        )
        self.alive[np.isin(self.pair, self.pair[cut])] = False
        if getattr(self, "Rgfc", None) is not None:
            self.gfc_clear(0.5 * (P1[cut] + P2[cut]))
        if self.kind == "ghost":
            rel = self.relax_near(list(np.unique(self.pair[cut])))
            extra = None
            if getattr(self, "vis_notch", False):
                xs = np.linspace(0.0, notch, max(2, int(np.ceil(notch / 0.25)) + 1))
                extra = np.column_stack(
                    [
                        xs[:-1],
                        np.full(len(xs) - 1, W / 2),
                        xs[1:],
                        np.full(len(xs) - 1, W / 2),
                    ]
                )
            if (
                self.ca_update(list(np.unique(self.pair[cut])), extra=extra) == 0
                and rel
            ):
                self.build()
        self.update_K()

    def build(self):
        nb, N = (self.nb, self.N)
        rows, cols, vals = ([], [], [])
        for b in range(nb):
            i, nv = (self.I[b], self.n[b])
            rows += [b, b]
            cols += [2 * i, 2 * i + 1]
            vals += [-nv[0], -nv[1]]
            if self.J[b] >= 0:
                j = self.J[b]
                rows += [b, b]
                cols += [2 * j, 2 * j + 1]
                vals += [nv[0], nv[1]]
            else:
                for m_, w_ in zip(self.sup[b], self.w[b]):
                    if w_ != 0.0:
                        rows += [b, b]
                        cols += [2 * m_, 2 * m_ + 1]
                        vals += [w_ * nv[0], w_ * nv[1]]
        self.B = sp.csr_matrix((vals, (rows, cols)), shape=(nb, 2 * N))
        self.update_K()

    def pair_ghosts(self, rmax=0.6, mode="max", cosmin=0.9):
        """bond-pair reconstitution across a non-matching boundary: ghost half-bond b (owner i in grain g, site s in grain
        h) <-> ghost half-bond c owned by the grain-h particle j nearest to s, pointing back (n_c . n_b < -0.9) with its
        site nearest to x_i (< rmax), mutual best match; a pair breaks as one bond (as the two directed halves of a bulk
        bond).  mode 'mean': the pair's breaking ratio is the mean of the two halves (two independent interpolations of
        the same connection averaged).  Returns the number of pairs."""
        gh = np.nonzero(self.J < 0)[0]
        site = self.X[self.I] + self.n * self.L0[:, None]
        own = {}
        for b in gh:
            own.setdefault(int(self.I[b]), []).append(b)
        trees = {
            g: (cKDTree(self.X[self.gid == g]), np.nonzero(self.gid == g)[0])
            for g in np.unique(self.gid)
        }
        best = {}
        for b in gh:
            sup = self.sup[b]
            h = int(np.bincount(self.gid[sup]).argmax())
            _, q = trees[h][0].query(site[b])
            j = int(trees[h][1][q])
            g = int(self.gid[self.I[b]])
            cands = [
                c
                for c in own.get(j, [])
                if int(np.bincount(self.gid[self.sup[c]]).argmax()) == g
                and self.n[c] @ self.n[b] < -cosmin
            ]
            if not cands:
                continue
            r = [np.linalg.norm(site[c] - self.X[self.I[b]]) for c in cands]
            k = int(np.argmin(r))
            if r[k] < rmax:
                best[b] = cands[k]
        self.partner = np.full(self.nb, -1)
        npair = 0
        for b, c in best.items():
            if best.get(c) == b and b < c:
                self.pair[c] = self.pair[b] = b
                self.partner[b], self.partner[c] = (c, b)
                npair += 1
        if mode == "mean_all":
            for b, c in best.items():
                if self.partner[b] < 0:
                    self.partner[b] = c
            mode = "mean"
        self.pair_mode = mode
        return npair

    def set_softening(self, m_soft, nt, a=0.5):
        """bilinear bond law (peak at the critical elongation e_c, linear softening to zero force at m_soft e_c) as a
        saw-tooth (sequentially linear analysis): tooth s has secant stiffness a^s k and triggers where a^s k e meets the
        softening line, e_s = m e_c / ((m - 1) a^s + 1); after nt teeth the bond is removed.  m_soft = 1 or nt = 0: brittle.
        """
        self.soft_m, self.soft_nt, self.soft_a = (float(m_soft), int(nt), float(a))

    def damage(self, b):
        """one saw-tooth event on bond (pair) b; returns True if the bond (pair) is now removed"""
        q = self.pair == self.pair[b]
        if self.soft_nt <= 0 or self.soft_m <= 1.0 or self.stage[b] + 1 >= self.soft_nt:
            self.alive[q] = False
            return True
        st = self.stage[b] + 1
        ph = self.soft_a**st
        self.stage[q] = st
        self.phi[q] = ph
        self.efac[q] = self.soft_m / ((self.soft_m - 1.0) * ph + 1.0)
        return False

    def update_K(self):
        a = self.alive * self.phi
        Sm = sp.csr_matrix((a, (self.I, self.rb)), shape=(self.N, self.nb))
        Tm = sp.csr_matrix((a * self.tb, (self.I, self.rb)), shape=(self.N, self.nb))
        self.H = (sp.diags(a * self.kb) + 0.5 * (Tm.T @ Sm + Sm.T @ Tm)).tocsr()
        if getattr(self, "coupling", "energy") == "force" and self.kind == "ghost":
            Fop = (
                sp.diags(a)
                @ (
                    sp.diags(self.kb)
                    + 0.5 * (Sm.T @ Tm)
                    + 0.5 * sp.diags(self.tb) @ (Sm.T @ Sm)
                )
            ).tocsr()
            Rm = sp.csr_matrix(
                (np.ones(self.nb), (self.rb, self.rev)), shape=(self.nb, self.nb)
            )
            A = sp.csr_matrix(
                (
                    np.r_[-self.n[:, 0], -self.n[:, 1]],
                    (np.r_[2 * self.I, 2 * self.I + 1], np.r_[self.rb, self.rb]),
                ),
                shape=(2 * self.N, self.nb),
            )
            self.K = (A @ (sp.identity(self.nb) + Rm) @ Fop @ self.B).tocsr()
        else:
            self.K = (self.B.T @ self.H @ self.B).tocsr()
            if getattr(self, "RG", None) is not None:
                self.K = (self.K - self.RG @ self.B).tocsr()
        Kff = self.K[self.free][:, self.free].tocsc()
        Kff = Kff + sp.identity(Kff.shape[0], format="csc") * (
            1e-12 * abs(Kff.diagonal()).max()
        )
        self.lu = spla.splu(Kff)
        self.Kfc = self.K[self.free][:, self.fixed].tocsc()

    def solve_unit(self):
        u = self.ug.copy()
        u[self.free] = self.lu.solve(-(self.Kfc @ self.ug[self.fixed]))
        return u

    def enable_gfc(self, margin=3.5):
        """ghost-force correction of the energy coupling (as gbp2d.Poly2D.enable_gfc), from the INTACT model with its
        final weights: R_i (2 x 3) = force on particle i per unit uniform strain [exx, eyy, exy]; kept only for free
        particles with intact stars away from the free edges (residuals there are surface physics, not GB error);
        f -= R_i eps_i, eps_i = least-squares strain of i from its own bonds.  gfc_clear() drops it near cracks.
        """
        assert self.kind == "ghost" and getattr(self, "coupling", "energy") == "energy"
        N, nb = (self.N, self.nb)
        self.RG = None
        self.update_K()
        R = np.zeros((N, 2, 3))
        for k, E in enumerate(
            (
                np.array([[1.0, 0], [0, 0]]),
                np.array([[0, 0], [0, 1.0]]),
                np.array([[0, 1.0], [1.0, 0]]),
            )
        ):
            R[:, :, k] = (self.K @ (self.X @ E.T).ravel()).reshape(-1, 2)
        R[~interior_complete(self, margin)] = 0.0
        v = self.L0[:, None] * np.column_stack(
            [self.n[:, 0] ** 2, self.n[:, 1] ** 2, 2 * self.n[:, 0] * self.n[:, 1]]
        )
        M = np.zeros((N, 3, 3))
        np.add.at(M, self.I, v[:, :, None] * v[:, None, :])
        self._gcoef = np.einsum("bij,bj->bi", np.linalg.pinv(M)[self.I], v)
        self.Rgfc = R
        self._gfc_rebuild()

    def _gfc_rebuild(self):
        c = np.einsum("bck,bk->bc", self.Rgfc[self.I], self._gcoef)
        self.RG = sp.csr_matrix(
            (
                c.ravel(),
                (
                    np.column_stack([2 * self.I, 2 * self.I + 1]).ravel(),
                    np.repeat(self.rb, 2),
                ),
            ),
            shape=(2 * self.N, self.nb),
        )
        self.update_K()

    def gfc_clear(self, pts, r=4.5):
        """no correction within r of the given points (crack faces: their residual is surface physics, and their ghost
        weights / stars have changed).  Returns True if anything changed (K rebuilt)."""
        if getattr(self, "Rgfc", None) is None or not len(pts):
            return False
        if not hasattr(self, "_ptree"):
            self._ptree = cKDTree(self.X)
        hit = set()
        for lst in self._ptree.query_ball_point(np.atleast_2d(pts), r):
            hit.update(lst)
        hit = np.array(sorted(hit), int)
        hit = hit[np.abs(self.Rgfc[hit]).max(axis=(1, 2)) > 0] if len(hit) else hit
        if not len(hit):
            return False
        self.Rgfc[hit] = 0.0
        c = np.einsum("bck,bk->bc", self.Rgfc[self.I], self._gcoef)
        self.RG = sp.csr_matrix(
            (
                c.ravel(),
                (
                    np.column_stack([2 * self.I, 2 * self.I + 1]).ravel(),
                    np.repeat(self.rb, 2),
                ),
            ),
            shape=(2 * self.N, self.nb),
        )
        return True

    def relax_near(self, broken_pairs):
        """damage-neighbourhood relaxation: optimised (extrapolation-type) weights are accurate for uniform fields but
        not in a crack-tip field; ghost sites within relax_R of an endpoint of a newly broken pair return to their MLS
        weights (both are linear exact).  Returns the list of relaxed ghost bonds."""
        if self.relax_R <= 0 or self.kind != "ghost":
            return []
        pts = []
        for p_ in broken_pairs:
            for q in np.nonzero(self.pair == p_)[0]:
                pts.append(self.X[self.I[q]])
                pts.append(self.X[self.I[q]] + self.n[q] * self.L0[q])
        if not pts:
            return []
        if not hasattr(self, "_gtree"):
            self._gb = np.nonzero(self.J < 0)[0]
            self._gtree = cKDTree(self.X[self.I[self._gb]] + self.n[self._gb])
        hit = set()
        for lst in self._gtree.query_ball_point(np.array(pts), self.relax_R):
            hit.update(lst)
        out = []
        for h in hit:
            b = self._gb[h]
            if not np.array_equal(self.w_static[b], self.w_mls[b]):
                self.w_static[b] = self.w_mls[b].copy()
                self.w[b] = self.w_mls[b].copy()
                out.append(b)
        return out

    def vis_update(self, broken_pairs, facet=None, reach=3.5, extra=None):
        """visibility criterion (meshfree cracks): every broken bond carries a crack facet = the segment of length
        facet * L0 perpendicular to the bond through its midpoint; a candidate support whose line of sight to a ghost site
        crosses any facet is dropped; MLS on the visible candidates (fewer than 3 non-collinear -> nearest visible
        particle; none visible -> nearest, or with vis_kill: the half-bond is removed -- the other grain has no material
        on this side of the crack, the owner is a surface particle in that direction).  `extra`: further facets (k, 4),
        e.g. the notch.  Updates the ghost sites within `reach` of a new facet."""
        if self.kind != "ghost":
            return 0
        facet = getattr(self, "vis_facet", 0.7) if facet is None else facet
        if not hasattr(self, "_facets"):
            self._facets = np.zeros((0, 4))
            self._gb = np.nonzero(self.J < 0)[0]
            self._gsite = (
                self.X[self.I[self._gb]] + self.n[self._gb] * self.L0[self._gb][:, None]
            )
            self._gtree = cKDTree(self._gsite)
        new = []
        for p_ in broken_pairs:
            q = np.nonzero(self.pair == p_)[0][0]
            c = self.X[self.I[q]] + 0.5 * self.n[q] * self.L0[q]
            t = np.array([-self.n[q, 1], self.n[q, 0]]) * 0.5 * facet * self.L0[q]
            new.append(np.r_[c - t, c + t])
        if extra is not None and len(extra):
            new += [f for f in np.asarray(extra, float)]
        if not new:
            return 0
        new = np.array(new)
        self._facets = np.vstack([self._facets, new])
        ftree = cKDTree(0.5 * (self._facets[:, :2] + self._facets[:, 2:]))
        hit = set()
        for lst in self._gtree.query_ball_point(0.5 * (new[:, :2] + new[:, 2:]), reach):
            hit.update(lst)
        ch = 0
        for h in hit:
            b = self._gb[h]
            if not self.alive[b]:
                continue
            site = self._gsite[h]
            cnd = self.cand[b]
            near = ftree.query_ball_point(site, reach + 1.0)
            F = self._facets[near]
            P = self.X[cnd]
            vis = np.ones(len(cnd), bool)
            if len(F):
                A, Bf = (F[:, :2], F[:, 2:])
                d1 = P[:, None, :] - site
                d2 = (Bf - A)[None, :, :]
                cross = lambda u, v: u[..., 0] * v[..., 1] - u[..., 1] * v[..., 0]
                den = cross(d1, d2)
                w0 = A[None, :, :] - site
                with np.errstate(divide="ignore", invalid="ignore"):
                    ta = cross(w0, d2) / den
                    tb = cross(w0, d1) / den
                blk = (
                    (np.abs(den) > 1e-12)
                    & (ta > 1e-09)
                    & (ta < 1 - 1e-09)
                    & (tb >= 0)
                    & (tb <= 1)
                )
                vis = ~blk.any(1)
            if vis.all():
                wn = self.w_static[b]
            else:
                wn = np.zeros(len(cnd))
                Y = P[vis] - site
                if (
                    vis.sum() >= 3
                    and np.linalg.matrix_rank(
                        np.column_stack([np.ones(vis.sum()), Y]), tol=1e-06
                    )
                    == 3
                ):
                    wn[vis] = self._wfun(Y)
                elif not vis.any() and getattr(self, "vis_kill", False):
                    self.alive[b] = False
                    ch += 1
                    self.killed = getattr(self, "killed", 0) + 1
                    continue
                else:
                    dist = np.linalg.norm(P - site, axis=1)
                    k0 = (
                        int(np.argmin(np.where(vis, dist, np.inf)))
                        if vis.any()
                        else int(np.argmin(dist))
                    )
                    wn[k0] = 1.0
            if not np.allclose(wn, self.w[b]):
                self.w[b] = wn
                ch += 1
        if ch:
            self.build()
        return ch

    def ca_update(self, broken_pairs, extra=None):
        """ghost half-bonds whose candidate sets contain an endpoint of a broken real pair: re-partition, MLS on the
        component of the nearest candidate (fewer than 3 non-collinear -> nearest particle alone).  Returns #changed.
        """
        if self.kind != "ghost":
            return 0
        if getattr(self, "use_vis", False):
            return self.vis_update(broken_pairs, extra=extra)
        ends = set()
        for p_ in broken_pairs:
            b = np.nonzero(self.pair == p_)[0]
            for q in b:
                ends.add(int(self.I[q]))
                if self.J[q] >= 0:
                    ends.add(int(self.J[q]))
        if not ends:
            return 0
        if not hasattr(self, "_cand_of"):
            self._cand_of = {}
            for b, c in enumerate(self.cand):
                if c is not None:
                    for m_ in c:
                        self._cand_of.setdefault(int(m_), []).append(b)
            real = self.J >= 0
            self._adj_pairs = {}
            for q in np.nonzero(real)[0]:
                self._adj_pairs.setdefault(int(self.I[q]), []).append(q)
        ch = 0
        todo = set()
        for e in ends:
            todo.update(self._cand_of.get(e, []))
        for b in todo:
            if not self.alive[b]:
                continue
            c = self.cand[b]
            idx = {int(m_): t for t, m_ in enumerate(c)}
            rr, cc = ([], [])
            for t, m_ in enumerate(c):
                for q in self._adj_pairs.get(int(m_), []):
                    if self.alive[q] and int(self.J[q]) in idx:
                        rr.append(t)
                        cc.append(idx[int(self.J[q])])
            A = sp.csr_matrix((np.ones(len(rr)), (rr, cc)), shape=(len(c), len(c)))
            ncomp, lab = connected_components(A, directed=False)
            site = self.X[self.I[b]] + self.n[b]
            dist = np.linalg.norm(self.X[c] - site, axis=1)
            k0 = int(np.argmin(dist))
            keep = lab == lab[k0]
            if keep.all():
                wn = self.w_static[b]
            else:
                Y = self.X[c[keep]] - site
                wn = np.zeros(len(c))
                if (
                    keep.sum() >= 3
                    and np.linalg.matrix_rank(
                        np.column_stack([np.ones(keep.sum()), Y]), tol=1e-06
                    )
                    == 3
                ):
                    wn[keep] = mls_weights(Y, 1.0)
                else:
                    wn[k0] = 1.0
            if not np.allclose(wn, self.w[b]):
                self.w[b] = wn
                ch += 1
        if ch:
            self.build()
        return ch


def wopt(model, damp=0.03, rho=100.0, iters=30000, positive=False):
    """generalised patch test weights (as gbp_wopt.py) for the ghost energy coupling of Frac2D: with linear-exact weights
    the half-bond 'forces' F = H dl of a uniform strain are known and f = B^T F is linear in the weights.
    """
    m = model
    gh = np.nonzero(m.J < 0)[0]
    wb, wpart, w0, Ys = ([], [], [], [])
    for b in gh:
        site = m.X[m.I[b]] + m.n[b]
        for m_, w_ in zip(m.sup[b], m.w[b]):
            wb.append(b)
            wpart.append(m_)
            w0.append(w_)
            Ys.append(m.X[m_] - site)
    wb, wpart, w0, Y = (np.array(wb), np.array(wpart), np.array(w0), np.array(Ys))
    nw = len(w0)
    gbs, ginv = np.unique(wb, return_inverse=True)
    Ca = sp.csr_matrix(
        (
            np.r_[np.ones(nw), Y[:, 0], Y[:, 1]],
            (np.r_[3 * ginv, 3 * ginv + 1, 3 * ginv + 2], np.tile(np.arange(nw), 3)),
        ),
        shape=(3 * len(gbs), nw),
    )
    da = np.tile([1.0, 0, 0], len(gbs))
    rb = m.rb
    real = m.J >= 0
    B0 = sp.csr_matrix(
        (
            np.r_[-m.n[:, 0], -m.n[:, 1], m.n[real, 0], m.n[real, 1]],
            (
                np.r_[rb, rb, rb[real], rb[real]],
                np.r_[2 * m.I, 2 * m.I + 1, 2 * m.J[real], 2 * m.J[real] + 1],
            ),
        ),
        shape=(m.nb, 2 * m.N),
    )
    sel = interior_complete(m)
    has_w = np.zeros(m.N, bool)
    has_w[wpart] = True
    blocks, rhs = ([], [])
    for E in (
        np.array([[1.0, 0], [0, 0]]),
        np.array([[0, 0], [0, 1.0]]),
        np.array([[0, 0.5], [0.5, 0]]),
    ):
        dl = m.L0 * np.einsum("bi,ij,bj->b", m.n, E, m.n)
        F = m.H @ dl
        r0 = (B0.T @ F).reshape(-1, 2)
        G = sp.csr_matrix(
            (
                np.r_[m.n[wb, 0] * F[wb], m.n[wb, 1] * F[wb]],
                (np.r_[2 * wpart, 2 * wpart + 1], np.tile(np.arange(nw), 2)),
            ),
            shape=(2 * m.N, nw),
        )
        keep = np.repeat(sel & (has_w | (np.abs(r0).max(1) > 1e-12)), 2)
        blocks.append(G[keep])
        rhs.append(-r0.ravel()[keep])
    C = sp.vstack([Ca] + blocks).tocsr()
    d = np.r_[da, np.concatenate(rhs)]
    rn = np.sqrt(np.asarray(C.multiply(C).sum(1)).ravel())
    ok = rn > 1e-12 * rn.max()
    ok[: Ca.shape[0]] = True
    C, d, rn = (C[ok], d[ok], rn[ok])
    sc = 1.0 / rn
    sc[: Ca.shape[0]] *= rho
    sol = spla.lsqr(
        sp.diags(sc) @ C,
        sc * (d - C @ w0),
        atol=1e-13,
        btol=1e-13,
        iter_lim=iters,
        damp=damp,
    )
    w = w0 + sol[0]
    order = np.argsort(ginv, kind="stable")
    ptr = np.searchsorted(ginv[order], np.arange(len(gbs) + 1))
    for g in range(len(gbs)):
        q = order[ptr[g] : ptr[g + 1]]
        P = np.vstack([np.ones(len(q)), Y[q].T])
        w[q] += P.T @ np.linalg.solve(P @ P.T, np.array([1.0, 0, 0]) - P @ w[q])
        if positive:
            lo = np.minimum(w0[q], 0.0)
            dw = w[q] - w0[q]
            neg = dw < -1e-15
            beta = (
                min(1.0, np.min((w0[q][neg] - lo[neg]) / -dw[neg]))
                if neg.any()
                else 1.0
            )
            w[q] = w0[q] + beta * dw
    pos = 0
    for b in gh:
        k = len(m.sup[b])
        m.w[b] = w[pos : pos + k].copy()
        m.w_static[b] = m.w[b].copy()
        pos += k
    m.build()


def interior_complete(m, margin=3.5):
    """free particles away from the free lateral edges whose bonds are all intact (not notch / crack faces)"""
    ok = m.free[0::2] & (m.X[:, 0] > margin) & (m.X[:, 0] < m.W - margin)
    dead_owner = np.zeros(m.N, bool)
    dead_owner[m.I[~m.alive]] = True
    dead_far = np.zeros(m.N, bool)
    r = ~m.alive & (m.J >= 0)
    dead_far[m.J[r]] = True
    return ok & ~dead_owner & ~dead_far


def patch_residual(m):
    out = []
    for E in (
        np.array([[1.0, 0], [0, 0]]),
        np.array([[0, 0], [0, 1.0]]),
        np.array([[0, 0.5], [0.5, 0]]),
    ):
        ua = (m.X @ E.T).ravel()
        f = (m.K @ ua).reshape(-1, 2)
        inn = interior_complete(m)
        out.append(np.abs(f[inn]).max() / m.kb.max())
    return out


def run(m, max_events=4000, drop=0.05, sep=0.01):
    """stops when the specimen is separated (structural stiffness R1 < sep * initial) -- after separation the floating
    pieces have ~zero stiffness and the solution is meaningless -- or the load fell below drop * peak.
    """
    Delta, peak, hist, broken, damaged = (0.0, 0.0, [], [], [])
    t0 = time.time()
    R10 = None
    for ev in range(max_events):
        u1 = m.solve_unit()
        dl1 = m.B @ u1
        R1 = (m.K @ u1)[2 * np.nonzero(m.top)[0] + 1].sum()
        R10 = R1 if R10 is None else R10
        if R1 < sep * R10:
            break
        r = np.where(m.alive & (dl1 > 0), dl1 / (m.dlc * m.efac), 0.0)
        if getattr(m, "pair_mode", "") == "mean":
            pp = m.partner >= 0
            pa = m.partner[pp]
            r = r.copy()
            r[pp] = np.where(m.alive[pa], 0.5 * (r[pp] + r[pa]), r[pp])
        b = int(np.argmax(r))
        if r[b] <= 0:
            break
        Dc = 1.0 / r[b]
        Delta = max(Delta, Dc)
        F = Delta * R1
        peak = max(peak, F)
        hist.append((Delta, F, int(m.gbb[b])))
        if F < drop * peak and ev > 10 and (m.soft_nt <= 0):
            break
        p_ = m.pair[b]
        removed = m.damage(b)
        (
            m.gfc_clear(m.X[m.I[b]][None] + 0.5 * m.n[b] * m.L0[b])
            if hasattr(m, "gfc_clear")
            else None
        )
        if not removed:
            damaged.append(b)
            m.update_K()
            continue
        broken.append(b)
        rel = m.relax_near([p_])
        if m.ca_update([p_]) == 0:
            if rel:
                m.build()
            else:
                m.update_K()
    return dict(
        hist=hist,
        broken=broken,
        damaged=damaged,
        peak=peak,
        t=time.time() - t0,
        events=ev + 1,
    )
