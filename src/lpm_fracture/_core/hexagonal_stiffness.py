"""Match the hexagonal lattice energy to an in-plane stiffness tensor.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import math
import numpy as np

SQ3 = math.sqrt(3.0)
VP = SQ3 / 2
DIRS = np.radians([0.0, 60.0, 120.0])


def voigt_strains():
    return [
        np.array([[1, 0], [0, 0.0]]),
        np.array([[0, 0], [0, 1.0]]),
        np.array([[0, 0.5], [0.5, 0]]),
    ]


def energy_density(k, T, eps):
    """uniform strain, complete star in the crystal frame: W = U_i / V"""
    el = []
    tv = []
    for J in range(3):
        n = np.array([math.cos(DIRS[J]), math.sin(DIRS[J])])
        e = n @ eps @ n
        el += [e, e]
        tv += [T[J], T[J]]
    el, tv = (np.array(el), np.array(tv))
    kk = np.repeat(k, 2)
    return (0.5 * (kk * el**2).sum() + 0.5 * (tv * el).sum() * el.sum()) / VP


def stiffness_of(k, T):
    E = voigt_strains()
    C = np.zeros((3, 3))
    for a in range(3):
        for b in range(3):
            C[a, b] = (
                energy_density(k, T, E[a] + E[b])
                - energy_density(k, T, E[a])
                - energy_density(k, T, E[b])
            )
    return C


def params_from_C(C):
    """(k1..3, T1..3) such that stiffness_of = C (Voigt 3x3: 11 22 12 66 16 26)"""
    idx = [(0, 0), (1, 1), (0, 1), (2, 2), (0, 2), (1, 2)]
    M = np.zeros((6, 6))
    for c in range(6):
        v = np.zeros(6)
        v[c] = 1.0
        Cc = stiffness_of(v[:3], v[3:])
        M[:, c] = [Cc[i, j] for i, j in idx]
    x = np.linalg.solve(M, np.array([C[i, j] for i, j in idx]))
    return (x[:3], x[3:])


def rotate_C2(C, th):
    """2D Voigt stiffness of a crystal whose frame is rotated by th (specimen = R(th) crystal)"""
    c, s = (math.cos(th), math.sin(th))
    Te = np.array(
        [
            [c * c, s * s, c * s],
            [s * s, c * c, -c * s],
            [-2 * c * s, 2 * c * s, c * c - s * s],
        ]
    )
    return Te.T @ C @ Te
