"""Elasticity.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np
from .bcc_lattice import BCCModel, params
from .grain_geometry import build_cloud, rotz, R_A
from .linear_energy import LPM3D

LZ = 3.0
VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def material(kind):
    if kind == "iso":
        C11, C12 = (236.0, 140.0)
        return dict(C11=C11, C12=C12, C44=0.5 * (C11 - C12))
    if kind == "fe":
        return dict(C11=236.0, C12=140.0, C44=116.0)
    raise ValueError(kind)


def cubic_C(C11, C12, C44):
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    np.fill_diagonal(C[:3, :3], C11)
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def rotate_C(C, Q):
    """Voigt stiffness rotated by Q (specimen = Q @ crystal)."""
    c = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            for p, q in {(i, j), (j, i)}:
                for r, s in {(k, l), (l, k)}:
                    c[p, q, r, s] = C[a, b]
    cr = np.einsum("ia,jb,kc,ld,abcd->ijkl", Q, Q, Q, Q, c)
    out = np.zeros((6, 6))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            out[a, b] = cr[i, j, k, l]
    return out


def bicrystal(theta_deg, Lx=40.0, Ly=40.0, shift=(0.0, 0.0, 0.0), rmerge=0.52):
    """tilt GB about z at y = Ly/2: grain 0 = R_A rotated by -theta/2, grain 1 by +theta/2 (+ rigid shift)."""
    f0 = rotz(-theta_deg / 2) @ R_A
    f1 = rotz(theta_deg / 2) @ R_A
    frames = [f0, f1]
    assign = lambda X: (X[:, 1] >= Ly / 2).astype(int)
    X, gid, nm = build_cloud(
        frames, assign, (0.0, 0.0), (Lx, Ly), LZ, rmerge=rmerge, shifts=[None, shift]
    )
    return (X, gid, nm, frames)


def polycrystal(seed, ng=6, L=50.0, rmerge=0.52):
    rng = np.random.default_rng(seed)
    seeds = rng.uniform(0, L, (ng, 2))
    angs = rng.uniform(0, 90, ng)
    frames = [rotz(a) @ R_A for a in angs]

    def assign(X):
        d = np.linalg.norm(X[:, None, :2] - seeds[None], axis=2)
        return np.argmin(d, 1)

    X, gid, nm = build_cloud(
        frames,
        assign,
        (0.0, 0.0),
        (L, L),
        LZ,
        rmerge=rmerge,
        shifts=[tuple(rng.uniform(0, 1, 3)) for _ in range(ng)],
    )
    return (X, gid, nm, frames)


def model(X, mat):
    p = params(mat["C11"], mat["C12"], mat["C44"], a=1.0, mapping="paper")
    bm = BCCModel(X, None, p, Lz=LZ)
    return (LPM3D(bm, p["Kn"], p["T"]), bm, p)
