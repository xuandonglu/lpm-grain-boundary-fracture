"""Elastic interface fields.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np

EPS0 = 0.001
MODES = {
    "xx": np.array([[1, 0, 0], [0, 0, 0], [0, 0, 0]]) * EPS0,
    "yy": np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]) * EPS0,
    "xy": np.array([[0, 0.5, 0], [0.5, 0, 0], [0, 0, 0]]) * EPS0,
}


def voigt(E):
    return np.array([E[0, 0], E[1, 1], E[2, 2], 2 * E[1, 2], 2 * E[0, 2], 2 * E[0, 1]])


def exact_bicrystal(Ebar, CA, CB):
    """displacement gradients HA, HB (plane strain, GB normal y, no z dependence): shared columns x (and z = 0),
    grain-specific column y. unknowns: d = dux/... see report. Returns HA, HB."""
    a, d = (Ebar[0, 0], Ebar[0, 1])

    def H(b, e, h):
        return np.array([[a, b, 0.0], [d, e, 0.0], [0.0, h, 0.0]])

    def sig(C, Hm):
        eps = 0.5 * (Hm + Hm.T)
        return C @ voigt(eps)

    def residual(x):
        HA, HB = (H(*x[:3]), H(*x[3:]))
        sA, sB = (sig(CA, HA), sig(CB, HB))
        eA, eB = (0.5 * (HA + HA.T), 0.5 * (HB + HB.T))
        r = [
            sA[1] - sB[1],
            sA[5] - sB[5],
            sA[3] - sB[3],
            0.5 * (eA[1, 1] + eB[1, 1]) - Ebar[1, 1],
            0.5 * (eA[0, 1] + eB[0, 1]) - Ebar[0, 1],
            0.5 * (eA[1, 2] + eB[1, 2]) - Ebar[1, 2],
        ]
        return np.array(r)

    r0 = residual(np.zeros(6))
    J = np.stack([residual(np.eye(6)[k]) - r0 for k in range(6)], 1)
    x = np.linalg.solve(J, -r0)
    return (H(*x[:3]), H(*x[3:]))
