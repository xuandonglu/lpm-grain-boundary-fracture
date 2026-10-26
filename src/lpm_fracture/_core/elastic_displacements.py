"""Elastic displacements.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np


def u_exact(P, HA, HB, ygb):
    x0 = np.array([0.0, ygb, 0.0])
    return np.where((P[:, 1] < ygb)[:, None], (P - x0) @ HA.T, (P - x0) @ HB.T)
