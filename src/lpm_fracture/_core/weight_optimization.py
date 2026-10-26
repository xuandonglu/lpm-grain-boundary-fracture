"""Moment-preserving regularization of ghost interpolation weights.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
from .elasticity import rotate_C
from .elastic_interface_fields import voigt

UNIT = [
    np.diag([1.0, 0, 0]),
    np.diag([0, 1.0, 0]),
    np.diag([0, 0, 1.0]),
    np.array([[0, 0, 0], [0, 0, 0.5], [0, 0.5, 0]]),
    np.array([[0, 0, 0.5], [0, 0, 0], [0.5, 0, 0]]),
    np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
]
LX = LY = 40.0
W_BAND = 2.6


def candidates(m, b):
    h = m.sgrain[b]
    d, jl = m.trees[h].query(m._rel(m.site[b]), k=m.nmls)
    return m.idx[h][jl]


def min_image(m, Y):
    Y = Y.copy()
    Y[:, 2] -= m.Lz * np.round(Y[:, 2] / m.Lz)
    return Y


def moment_rows(Y, order):
    """consistency matrix of a support cloud Y (nc, 3) relative to the site: rows 1, y (order 1) [+ the 6 quadratic monomials
    y_a y_b (order 2)]; A w = e_1 <=> the interpolation reproduces polynomials of that order exactly
    """
    rows = [np.ones(len(Y)), Y[:, 0], Y[:, 1], Y[:, 2]]
    if order >= 2:
        rows += [
            Y[:, 0] ** 2,
            Y[:, 1] ** 2,
            Y[:, 2] ** 2,
            Y[:, 0] * Y[:, 1],
            Y[:, 0] * Y[:, 2],
            Y[:, 1] * Y[:, 2],
        ]
    return np.vstack(rows)


def base_weights(m, c, s, order=1, h=None):
    """interpolation weights of site s on the candidate supports c and the null space of the consistency matrix.
    order 1: MLS weights (as GhostLPM).  order 2: the MLS weights corrected to quadratic exactness with the smallest
    Gaussian-weighted change, min sum dw^2 / phi subject to A (w + dw) = e1 (a quadratic-MLS-type interpolant that keeps the
    near supports dominant); falls back to order 1 if the cloud does not span the quadratics.
    """
    wm = dict(m._mls(c, s))
    w = np.array([wm.get(q, 0.0) for q in c])
    Y = min_image(m, m.X[c] - s)
    if order >= 2:
        A = moment_rows(Y, 2)
        sv = np.linalg.svd(A, compute_uv=False)
        if sv[-1] > 1e-06 * sv[0]:
            phi = np.exp(-((np.linalg.norm(Y, axis=1) / (h or m.hmls)) ** 2)) + 0.001
            e1 = np.zeros(A.shape[0])
            e1[0] = 1.0
            w = w + phi * (A.T @ np.linalg.solve(A * phi @ A.T, e1 - A @ w))
        else:
            order = 1
    A = moment_rows(Y, order)
    _, sv, vt = np.linalg.svd(A)
    rk = int((sv > 1e-10 * sv[0]).sum())
    return (w, vt[rk:].T, order)


def set_weights(m, wdict):
    """rebuild B0 / Gj of the GhostLPM with the given ghost-bond weights {b: (cand, w)}"""
    rows_B, cols_B, vals_B, rG, cG, vG = ([], [], [], [], [], [])
    for b in range(m.nb):
        i = m.i[b]
        sup = [(m.j[b], 1.0)] if not m.ghost[b] else list(zip(*wdict[b]))
        for q, w in sup:
            rows_B += [b, b, b]
            cols_B += [3 * q, 3 * q + 1, 3 * q + 2]
            vals_B += list(-w * m.n[b])
            rG.append(b)
            cG.append(2 * q + m.s[b])
            vG.append(w)
        rows_B += [b, b, b]
        cols_B += [3 * i, 3 * i + 1, 3 * i + 2]
        vals_B += list(m.n[b])
    m.B0 = sp.csr_matrix((vals_B, (rows_B, cols_B)), shape=(m.nb, 3 * m.N))
    m.Gj = sp.csr_matrix((vG, (rG, cG)), shape=(m.nb, 2 * m.N))


DIRECT_MAX = 25000
LSQR_PC = True
LSQR_ITS = 50000


def optimise_weights_gen(
    m, eq_mask, fields, mu=1e-06, rel_reg=None, order=1, per_owner=False, diag=False
):
    """generalised patch test: the energy model's internal forces vanish for the given exact piecewise-affine fields.
    fields: list of callables u(P) -> (len(P), 3) (z-independent, continuous at the GB).  Interpolation supports lie in the
    grain that owns the site, so the interpolated site displacement is exact for every weight set that is linear exact;
    the bond elongations of the exact field then do not depend on the weights and the conditions are linear in them.
    Exactness kept through the null-space parametrisation; regularised least squares for the patch rows.
    order 2: base weights quadratic exact (base_weights) and the null space of the quadratic consistency matrix, so the
    optimised weights stay quadratic exact (second-order accurate site displacements in strain-gradient fields).
    per_owner: fields are callables u(P, own) -- the target displacement at points P evaluated in the local field of the
    owner particles own (polycrystals: the exact bicrystal field of the owner's nearest boundary facet, poly_facet_fields);
    every bond's elongation is then taken in its owner's field, so a particle's own bonds are mutually consistent.
    """
    gb = np.nonzero(m.ghost)[0]
    nf = len(fields)
    sites = m.X[m.i] - m.n * m.d0[:, None]
    if per_owner:
        dL = np.stack(
            [
                np.einsum("bi,bi->b", m.n, f(m.X[m.i], m.i) - f(sites, m.i))
                for f in fields
            ],
            1,
        )
    else:
        dL = np.stack(
            [np.einsum("bi,bi->b", m.n, f(m.X)[m.i] - f(sites)) for f in fields], 1
        )
    D = np.zeros((m.N, 2, nf))
    np.add.at(D, (m.i, m.s), dL)
    qb = m.Kn[:, None] * dL + m.T * D[m.i, m.s]
    cand, w0, Nb = ({}, {}, {})
    nlow = 0
    for b in gb:
        c = candidates(m, b)
        cand[b] = c
        w0[b], Nb[b], o = base_weights(m, c, m.site[b], order)
        nlow += o < order
    if order >= 2 and nlow:
        print(
            "    %d of %d ghost bonds fell back to linear exactness" % (nlow, len(gb)),
            flush=True,
        )
    nr = 3 * nf
    f0 = np.zeros((m.N, 3, nf))
    np.add.at(f0, m.i, m.n[:, :, None] * qb[:, None, :])
    real = np.nonzero(~m.ghost)[0]
    np.add.at(f0, m.j[real], -m.n[real][:, :, None] * qb[real][:, None, :])
    for b in gb:
        np.add.at(
            f0,
            cand[b],
            -w0[b][:, None, None] * (m.n[b][:, None] * qb[b][None, :])[None],
        )
    eqp = np.nonzero(eq_mask)[0]
    eqrow = -np.ones(m.N, int)
    eqrow[eqp] = np.arange(len(eqp))
    zoff, off = ({}, 0)
    for b in gb:
        zoff[b] = off
        off += Nb[b].shape[1]
    rows, cols, vals = ([], [], [])
    for b in gb:
        coef = -(m.n[b][:, None] * qb[b][None, :]).reshape(-1)
        for kk, q in enumerate(cand[b]):
            if eqrow[q] < 0:
                continue
            rr = eqrow[q] * nr + np.arange(nr)
            blk = np.outer(coef, Nb[b][kk])
            for jcol in range(blk.shape[1]):
                rows += list(rr)
                cols += [zoff[b] + jcol] * nr
                vals += list(blk[:, jcol])
    P = sp.csr_matrix((vals, (rows, cols)), shape=(len(eqp) * nr, off))
    r0 = f0[eqp].reshape(-1)
    scale = float(np.asarray(P.multiply(P).sum(0)).ravel().mean())
    lsq_info = {}
    if off <= DIRECT_MAX:
        PtP = (P.T @ P).tocsc()
        z = spla.spsolve(PtP + mu * scale * sp.identity(off, format="csc"), -(P.T @ r0))
    elif not LSQR_PC:
        out = spla.lsqr(
            P, -r0, damp=np.sqrt(mu * scale), atol=1e-10, btol=1e-10, iter_lim=20000
        )
        z = out[0]
        lsq_info = {"lsqr_stop": int(out[1]), "lsqr_its": int(out[2])}
    else:
        A = sp.vstack([P, np.sqrt(mu * scale) * sp.identity(off, format="csr")]).tocsc()
        dcol = np.sqrt(np.asarray(A.multiply(A).sum(0)).ravel())
        dcol[dcol == 0] = 1.0
        out = spla.lsqr(
            A @ sp.diags(1.0 / dcol),
            np.r_[-r0, np.zeros(off)],
            atol=1e-12,
            btol=1e-12,
            iter_lim=LSQR_ITS,
        )
        z = out[0] / dcol
        lsq_info = {"lsqr_stop": int(out[1]), "lsqr_its": int(out[2])}
    res = P @ z + r0
    res_full, r0_full = (res.copy(), r0.copy())
    wdict = {
        b: (cand[b], w0[b] + Nb[b] @ z[zoff[b] : zoff[b] + Nb[b].shape[1]]) for b in gb
    }
    allw = np.concatenate([v[1] for v in wdict.values()])
    reach = np.asarray(abs(P).sum(1)).ravel() > 0
    r0, res = (r0[reach], res[reach])
    info = {
        "free_weights": off,
        "rows": int(P.shape[0]),
        "resid_mls": float(np.abs(r0).max()),
        "resid_opt": float(np.abs(res).max()),
        "resid_opt_rms": float(np.sqrt(np.mean(res**2))),
        "w_min": float(allw.min()),
        "w_max": float(allw.max()),
        "mu": mu,
        "order": order,
        **lsq_info,
    }
    if diag:
        info.update(
            {
                "_eqp": eqp,
                "_res": res_full.reshape(len(eqp), nr),
                "_res0": r0_full.reshape(len(eqp), nr),
            }
        )
    print(
        "    generalised target (%d fields), mu %.0e: residual %.2e -> %.2e (rms %.2e) | w in [%.2f, %.2f]"
        % (
            nf,
            mu,
            info["resid_mls"],
            info["resid_opt"],
            info["resid_opt_rms"],
            info["w_min"],
            info["w_max"],
        ),
        flush=True,
    )
    return (wdict, info)


def poly_facet_fields(X, gid, seeds, frames, C0, modes=None):
    """target fields for a columnar Voronoi polycrystal (z-periodic): every particle is given the facet between its grain a
    and the grain b of its nearest Voronoi facet (2026-10-13: was the nearest other seed, which is not the nearest facet) (normal n_ab = (s_b - s_a) / |..|, through the midpoint); for each in-plane far-field mode the
    exact piecewise-uniform bicrystal solution of that facet (exact_bicrystal_n, elastic constants rotated into the two grain
    frames).  Returns callables u(P, own) for optimise_weights_gen(..., per_owner=True) and the number of facets used.
    """
    from .elasticity import rotate_C

    modes = modes or [
        np.diag([1.0, 0, 0]),
        np.diag([0, 1.0, 0]),
        np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]),
    ]
    seeds = np.asarray(seeds, float)
    d2 = ((X[:, None, :2] - seeds[None, :, :]) ** 2).sum(-1)
    da = d2[np.arange(len(X)), gid]
    sep = np.sqrt(((seeds[gid][:, None, :] - seeds[None, :, :]) ** 2).sum(-1))
    with np.errstate(divide="ignore", invalid="ignore"):
        fd = (d2 - da[:, None]) / (2 * sep)
    fd[np.arange(len(X)), gid] = np.inf
    nbr = np.argmin(fd, 1)
    keys = sorted(set(zip(gid.tolist(), nbr.tolist())))
    fac = {}
    for a, b in keys:
        nv = np.r_[seeds[b] - seeds[a], 0.0]
        nv /= np.linalg.norm(nv)
        x0 = np.r_[0.5 * (seeds[a] + seeds[b]), 0.0]
        Hs = [
            exact_bicrystal_n(E, rotate_C(C0, frames[a]), rotate_C(C0, frames[b]), nv)
            for E in modes
        ]
        fac[a, b] = (nv, x0, Hs)
    kid = {k: i for i, k in enumerate(keys)}
    pk = np.array([kid[a, b] for a, b in zip(gid.tolist(), nbr.tolist())])

    def make(mi):

        def u(P, own):
            out = np.zeros((len(P), 3))
            ko = pk[own]
            for k in np.unique(ko):
                sel = ko == k
                nv, x0, Hs = fac[keys[k]]
                HA, HB = Hs[mi]
                R = P[sel] - x0
                side = (R @ nv >= 0.0)[:, None]
                out[sel] = np.where(side, R @ HB.T, R @ HA.T)
            return out

        return u

    return ([make(i) for i in range(len(modes))], len(keys))


def exact_bicrystal_n(Ebar, CA, CB, n):
    """piecewise-uniform exact bicrystal solution for a planar GB with unit normal n (n_z = 0) and a far-field strain Ebar
    whose z column vanishes (z-periodic models): E_A = Ebar - sym(a n)/2, E_B = Ebar + sym(a n)/2 (volume average Ebar,
    compatible jump), a from traction continuity (C_A:E_A - C_B:E_B).n = 0.  Returns displacement gradients H_A, H_B with a
    zero z column (z-independent displacements; rotation gauge) and H_B - H_A = a (x) n (continuous across the plane).
    """
    n = np.asarray(n, float) / np.linalg.norm(n)

    def ten(sv):
        return np.array(
            [[sv[0], sv[5], sv[4]], [sv[5], sv[1], sv[3]], [sv[4], sv[3], sv[2]]]
        )

    def trac(C, E):
        return ten(C @ voigt(E)) @ n

    def symdy(a):
        return 0.5 * (np.outer(a, n) + np.outer(n, a))

    rhs = trac(CA, Ebar) - trac(CB, Ebar)
    M = np.stack(
        [0.5 * (trac(CA, symdy(e)) + trac(CB, symdy(e))) for e in np.eye(3)], 1
    )
    a = np.linalg.solve(M, rhs)
    EA = Ebar - 0.5 * symdy(a)
    HA = EA.copy()
    for i in (0, 1):
        HA[i, 2] = 0.0
        HA[2, i] = 2 * EA[i, 2]
    assert abs(HA[2, 2]) < 1e-14
    HB = HA + np.outer(a, n)
    return (HA, HB)
