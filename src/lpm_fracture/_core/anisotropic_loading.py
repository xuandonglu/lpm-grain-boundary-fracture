"""Material constants and anisotropic crack-tip displacement fields.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np
from . import stroh_elasticity as gk

A_NEP = 2.835e-10
KCONV = 1000.0 * np.sqrt(A_NEP)
GPA_A = 1000000000.0 * A_NEP
NEP_C = (250.7, 123.8, 98.7)
GS = {
    "0K": {"100": 2.502, "110": 2.356, "111": 2.707, "112": 2.599},
    "300K": {"100": 2.39, "110": 2.26},
}


def stroh_frame(C, R):
    Ct = gk.C4(*C, R)
    p, A, B = gk.stroh(Ct)
    return (p, A, B, Ct)


def kfield_disp(C, R, x, y, k):
    """displacements (n,3) of the anisotropic near-tip field, crack along -x, tip at origin, k = (K_II, K_I, K_III)
    [GPa sqrt(a)], x,y [a].  u = sqrt(2r/pi) Re[A <sqrt(cos th + p_a sin th)> B^-1 k].
    """
    p, A, B, _ = stroh_frame(C, R)
    r = np.hypot(x, y)
    th = np.arctan2(y, x)
    z = np.cos(th)[:, None] + p[None, :] * np.sin(th)[:, None]
    q = np.linalg.solve(B, np.asarray(k, complex))
    u = np.real(np.sqrt(z) * q[None, :] @ A.T)
    return np.sqrt(2 * r / np.pi)[:, None] * u


def lambda22(C, R):
    L, _, _, _ = gk.Lam(gk.C4(*C, R))
    return L
