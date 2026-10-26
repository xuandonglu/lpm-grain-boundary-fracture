"""Linear energy.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla


class LPM3D:

    def __init__(self, bm, Kn_shell, T):
        """bm: bc_core.BCCModel (geometry only)."""
        self.X, self.N = (bm.X, bm.N)
        self.i, self.j, self.s = (bm.I, bm.J, bm.sh)
        self.n, self.d0, self.uid = (bm.n, bm.L0, bm.pair)
        self.nb = len(self.i)
        self.kn0 = np.asarray(Kn_shell, float)[self.s]
        self.tv0 = np.full(self.nb, float(T))
        self.Kn, self.Tv = (self.kn0.copy(), self.tv0.copy())
        self.w = np.ones(self.nb)
        self.V = bm.p["V"]
        self.Lz = bm.Lz

    def coordination(self):
        c = np.zeros((self.N, 2), np.int64)
        np.add.at(c, (self.i, self.s), 1)
        return c

    def matrices(self):
        nb, N = (self.nb, self.N)
        r = np.arange(nb)
        cols_i = np.stack([3 * self.i, 3 * self.i + 1, 3 * self.i + 2], 1)
        cols_j = np.stack([3 * self.j, 3 * self.j + 1, 3 * self.j + 2], 1)
        B = sp.csr_matrix(
            (
                np.concatenate([self.n.ravel(), -self.n.ravel()]),
                (
                    np.concatenate([np.repeat(r, 3), np.repeat(r, 3)]),
                    np.concatenate([cols_i.ravel(), cols_j.ravel()]),
                ),
            ),
            shape=(nb, 3 * N),
        )
        A = sp.csr_matrix(
            (self.n.ravel(), (np.repeat(r, 3), cols_i.ravel())), shape=(nb, 3 * N)
        )
        G = sp.csr_matrix((np.ones(nb), (2 * self.i + self.s, r)), shape=(2 * N, nb))
        Gi = sp.csr_matrix((np.ones(nb), (r, 2 * self.i + self.s)), shape=(nb, 2 * N))
        Gj = sp.csr_matrix((np.ones(nb), (r, 2 * self.j + self.s)), shape=(nb, 2 * N))
        return (B, A, G, Gi, Gj)

    def stiffness(self):
        B, A, G, Gi, Gj = self.matrices()
        Tv = sp.diags(self.Tv)
        M = sp.diags(2 * self.Kn) + 0.5 * (Gi + Gj) @ G @ Tv + 0.5 * Tv @ (Gi + Gj) @ G
        W = sp.diags(self.w)
        self._B, self._M = ((W @ B).tocsr(), (W @ M).tocsr())
        return (A.T @ W @ M @ W @ B).tocsr()

    def bond_force(self, u):
        dL = self._B @ u.ravel()
        return (self._M @ dL, dL)

    def stress(self, u):
        """Virial particle stress in Voigt order (xx, yy, zz, yz, xz, xy), with half weight per directed interaction (w = 1/2)."""
        F, _ = self.bond_force(u)
        c = 0.5 / self.V * self.d0 * F
        out = np.zeros((self.N, 6))
        for k, (a, b) in enumerate([(0, 0), (1, 1), (2, 2), (1, 2), (0, 2), (0, 1)]):
            np.add.at(out[:, k], self.i, c * self.n[:, a] * self.n[:, b])
        return out

    def solve_dirichlet(self, K, fixed, u_fixed, tol=1e-11):
        free = ~fixed
        u = np.zeros(3 * self.N)
        u[fixed] = u_fixed[fixed]
        Kff = K[free][:, free].tocsr()
        rhs = -K[free][:, fixed] @ u[fixed]
        d = Kff.diagonal()
        Mp = spla.LinearOperator(Kff.shape, matvec=lambda v: v / d)
        x, info = spla.cg(Kff, rhs, rtol=tol, maxiter=20000, M=Mp)
        if info != 0:
            x = spla.spsolve(Kff.tocsc(), rhs)
        u[free] = x
        return u.reshape(-1, 3)
