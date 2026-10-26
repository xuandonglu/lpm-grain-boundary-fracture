"""Hexagonal particle clouds, Voronoi grains and interpolation weights.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import math
import numpy as np
import scipy.sparse as sp
from scipy.spatial import cKDTree

SQ3 = math.sqrt(3.0)


def hex_points(theta, shift, lo, hi):
    L = 1.5 * max(hi[0] - lo[0], hi[1] - lo[1]) + 10
    n = int(L) + 4
    ii, jj = np.meshgrid(np.arange(-n, n + 1), np.arange(-n, n + 1), indexing="ij")
    P = np.stack(
        [(ii + 0.5 * (jj % 2)).ravel(), (SQ3 / 2 * jj).ravel()], 1
    ) + np.asarray(shift)
    c, s = (math.cos(theta), math.sin(theta))
    P = P @ np.array([[c, s], [-s, c]])
    m = (P[:, 0] >= lo[0]) & (P[:, 0] < hi[0]) & (P[:, 1] >= lo[1]) & (P[:, 1] < hi[1])
    return P[m]


def microstructure(W, ng, seed):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, W, (ng, 2))
    theta = rng.uniform(0, math.pi / 3, ng)
    shift = rng.uniform(0, 1, (ng, 2))
    return (seeds, theta, shift)


def assign(P, seeds):
    return np.argmin(((P[:, None, :] - seeds[None, :, :]) ** 2).sum(-1), 1)


def merge_overlaps(X, gid, thr):
    while True:
        pr = cKDTree(X).query_pairs(thr, output_type="ndarray")
        if not len(pr):
            return (X, gid)
        n = len(X)
        G = sp.csr_matrix((np.ones(len(pr)), (pr[:, 0], pr[:, 1])), shape=(n, n))
        nc, lab = sp.csgraph.connected_components(G, directed=False)
        cnt = np.bincount(lab, minlength=nc)
        C = np.zeros((nc, 2))
        np.add.at(C, lab, X)
        g2 = np.full(nc, -1)
        g2[lab] = gid
        X, gid = (C / cnt[:, None], g2)


def tri_weights(Y):
    """barycentric weights of the origin in the Delaunay triangle of the support offsets Y (k, 2) that contains it (or, if
    outside the hull, the triangle with the least negative barycentric coordinate): linear exact, interpolatory at nodes
    """
    from scipy.spatial import Delaunay

    tri = Delaunay(Y)
    s = int(tri.find_simplex(np.zeros((1, 2)))[0])
    T = tri.transform
    if s < 0:
        b = np.einsum("sij,sj->si", T[:, :2, :], -T[:, 2, :])
        bb = np.column_stack([b, 1 - b.sum(1)])
        mn = np.where(np.isfinite(bb).all(1), bb.min(1), -np.inf)
        s = int(np.argmax(mn))
    b = T[s, :2, :] @ -T[s, 2, :]
    w = np.zeros(len(Y))
    w[tri.simplices[s]] = np.r_[b, 1 - b.sum()]
    return w


def mls_weights_q(Y, h=1.2):
    """quadratic-exact MLS weights of the origin (basis 1, x, y, x^2, xy, y^2); linear MLS if singular"""
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


def mls_weights(Y, h=1.0):
    """linear-exact MLS weights of the origin from support offsets Y (k, 2)"""
    d = np.linalg.norm(Y, axis=1)
    w = np.exp(-((d / h) ** 2))
    P = np.column_stack([np.ones(len(Y)), Y])
    A = P.T @ (w[:, None] * P)
    return w * (P @ np.linalg.solve(A, np.array([1.0, 0.0, 0.0])))
