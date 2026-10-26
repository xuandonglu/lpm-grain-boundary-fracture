"""Stroh elasticity.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np

VOIGT = [(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]


def unit(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def C66(C11, C12, C44):
    C = np.zeros((6, 6))
    C[:3, :3] = C12
    np.fill_diagonal(C[:3, :3], C11)
    C[3, 3] = C[4, 4] = C[5, 5] = C44
    return C


def C4(C11, C12, C44, R=np.eye(3)):
    C = C66(C11, C12, C44)
    T = np.zeros((3, 3, 3, 3))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            for ii, jj in {(i, j), (j, i)}:
                for kk, ll in {(k, l), (l, k)}:
                    T[ii, jj, kk, ll] = C[a, b]
    return np.einsum("ai,bj,ck,dl,ijkl->abcd", R, R, R, R, T)


def to_voigt(T4):
    C = np.zeros((6, 6))
    for a, (i, j) in enumerate(VOIGT):
        for b, (k, l) in enumerate(VOIGT):
            C[a, b] = T4[i, j, k, l]
    return C


def stroh(Ct):
    Q = Ct[:, 0, :, 0]
    Rm = Ct[:, 0, :, 1]
    T = Ct[:, 1, :, 1]
    Ti = np.linalg.inv(T)
    N = np.block([[-Ti @ Rm.T, Ti], [Rm @ Ti @ Rm.T - Q, -Rm @ Ti]])
    w, v = np.linalg.eig(N)
    sel = np.argsort(-w.imag)[:3]
    return (w[sel], v[:3, sel], v[3:, sel])


def Lam(Ct):
    p, A, B = stroh(Ct)
    return (0.5 * np.real(1j * A @ np.linalg.inv(B)), p, A, B)


def stress(p, B, th, k):
    z = np.cos(th) + p * np.sin(th)
    Bi = np.linalg.inv(B)
    s2 = np.real(B @ np.diag(1 / np.sqrt(z)) @ Bi @ k)
    s1 = -np.real(B @ np.diag(p / np.sqrt(z)) @ Bi @ k)
    S = np.zeros((3, 3))
    S[:, 0] = s1
    S[:, 1] = s2
    S[0, :] = s1
    S[1, :] = s2
    return S


def frame(plane, front):
    ey = unit(plane)
    ez = unit(front)
    assert abs(ey @ ez) < 1e-12
    return np.vstack([np.cross(ey, ez), ey, ez])
