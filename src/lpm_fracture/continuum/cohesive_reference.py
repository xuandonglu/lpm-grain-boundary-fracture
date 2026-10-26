"""Q4 plane-strain cohesive FE comparison on the archived bicrystal geometry.

Mode-I symmetry is used only for this straight-cleavage load comparison. It is
NOT used to establish an independently predicted crack direction. The cohesive
plane has the rigid-separation potential in material_and_fem.py. Crack opening
at a moving material point is the continuation control; load lambda is solved,
not fitted. All bulk FE interior DOFs are statically condensed exactly.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.linalg as la
import scipy.sparse.linalg as sla
from .finite_elements import (
    inputs,
    Cohesive100,
    C2,
    grid,
    stiffness,
    kdisp,
    OUT,
    save_json,
)


class CondensedCohesive:

    def __init__(self, case, h):
        self.inp = inp = inputs(case)
        self.law = Cohesive100(inp["l"])
        self.h = h
        self.xs = np.linspace(0, inp["W"], round(inp["W"] / h) + 1)
        ys = np.linspace(
            inp["y0"], inp["y0"] + inp["H"] / 2, round(inp["H"] / 2 / h) + 1
        )
        self.xy, self.conn, self.dofs = grid(self.xs, ys)
        self.K = K = stiffness(self.xy, self.conn, self.dofs)
        self.bottom = np.arange(len(self.xs))
        band = (
            np.minimum(
                inp["W"] - self.xy[:, 0], inp["y0"] + inp["H"] / 2 - self.xy[:, 1]
            )
            < 1.5 - 1e-09
        )
        self.B = np.flatnonzero(np.repeat(band, 2))
        V = 2 * self.bottom + 1
        self.V = V[~np.isin(V, self.B)]
        self.xv = self.xy[self.V // 2, 0]
        self.I = np.setdiff1d(np.arange(K.shape[0]), np.r_[self.B, self.V])
        self.factor = sla.splu(K[self.I][:, self.I].tocsc())
        self.KIV = K[self.I][:, self.V].toarray()
        self.KIB = K[self.I][:, self.B].tocsc()
        self.S = K[self.V][:, self.V].toarray() - self.KIV.T @ self.factor.solve(
            self.KIV
        )
        self.S = (self.S + self.S.T) / 2
        self.force = None
        edge = np.flatnonzero(self.xs[:-1] >= inp["a0"] - 1e-08)
        N = np.array(
            [
                [(1 + 1 / np.sqrt(3)) / 2, (1 - 1 / np.sqrt(3)) / 2],
                [(1 - 1 / np.sqrt(3)) / 2, (1 + 1 / np.sqrt(3)) / 2],
            ]
        )
        self.edges = edge
        self.N = N
        self.w = np.diff(self.xs)[edge] / 2
        self.vmap = np.full(len(self.bottom), -1)
        self.vmap[self.V // 2] = np.arange(len(self.V))
        self.checks = dict(
            mesh_h=h,
            elements=len(self.conn),
            bulk_dofs=K.shape[0],
            interface_dofs=len(self.V),
            stiffness_symmetry=float(sla.norm(K - K.T) / sla.norm(K)),
            minimum_condensed_eigenvalue=float(
                la.eigvalsh(self.S, subset_by_index=[0, 0])[0]
            ),
        )
        strain = np.array([0.003, -0.001, 0.0007])
        aff = np.column_stack(
            [
                strain[0] * self.xy[:, 0] + 0.5 * strain[2] * self.xy[:, 1],
                strain[1] * self.xy[:, 1] + 0.5 * strain[2] * self.xy[:, 0],
            ]
        ).ravel()
        expected = 0.5 * inp["W"] * inp["H"] / 2 * (strain @ C2 @ strain)
        self.checks["affine_energy_relative_error"] = float(
            abs(0.5 * aff @ K @ aff / expected - 1)
        )

    def boundary(self, center):
        ub = kdisp(self.inp, self.xy[self.B // 2], center)
        self.ub = ub[np.arange(len(self.B)), self.B % 2]
        self.uIunit = -self.factor.solve(self.KIB @ self.ub)
        self.force = (
            np.asarray(self.K[self.V][:, self.B] @ self.ub).ravel()
            + self.KIV.T @ self.uIunit
        )

    def interface(self, v, lam):
        vb = np.zeros(len(self.bottom))
        vb[self.V // 2] = v
        bv = np.isin(2 * self.bottom + 1, self.B)
        vb[bv] = lam * kdisp(self.inp, self.xy[self.bottom[bv]], self.center)[:, 1]
        gap = 2 * vb[self.edges[:, None] + np.arange(2)] @ self.N.T
        en, tr, tg = self.law(gap)
        gv = self.S @ v + lam * self.force
        Hv = self.S.copy()
        for q in range(2):
            for a in range(2):
                ia = self.vmap[self.edges + a]
                ma = ia >= 0
                np.add.at(gv, ia[ma], self.w[ma] * tr[ma, q] * self.N[q, a])
                for b in range(2):
                    ib = self.vmap[self.edges + b]
                    mb = ma & (ib >= 0)
                    np.add.at(
                        Hv,
                        (ia[mb], ib[mb]),
                        2 * self.w[mb] * tg[mb, q] * self.N[q, a] * self.N[q, b],
                    )
        return (gv, Hv, float(np.sum(en * self.w[:, None])))

    def solve_front(self, front, initial=None, tol=1e-08):
        l1, l2 = self.law.l
        delta1 = np.sqrt((np.sqrt(0.75) + 3.8897 * l1) ** 2 - 0.5) - 0.5
        delta2 = 3.8897 * l2
        delta = min(delta1, delta2)
        ctrl = int(np.argmin(abs(self.xv - front)))
        front = float(self.xv[ctrl])
        self.center = (front - 0.5, self.inp["y0"])
        self.boundary(self.center)
        if initial is None:
            v = (
                0.5
                * delta
                * np.exp(
                    -np.maximum(self.xv - front, 0)
                    / max(self.law.ell_strength_AT2, 0.1)
                )
            )
            v += 0.025 * np.sqrt(np.maximum(front - self.xv, 0))
            z = np.r_[v, 1.0]
        else:
            z = initial.copy()
        z[ctrl] = delta / 2
        trace = []
        for it in range(100):
            g, H, ce = self.interface(z[:-1], z[-1])
            r = np.r_[g, 10 * (2 * z[ctrl] - delta)]
            err = float(np.max(abs(r)))
            trace.append(err)
            if err < tol:
                break
            J = np.zeros((len(z), len(z)))
            J[:-1, :-1] = H
            J[:-1, -1] = self.force
            J[-1, ctrl] = 20
            dz = la.solve(J, -r, assume_a="gen", check_finite=False)
            step = 1.0
            norm = la.norm(r)
            accepted = False
            for _ in range(25):
                trial = z + step * dz
                if trial[-1] > 0 and np.min(trial[:-1]) > -1e-06:
                    gt, _, _ = self.interface(trial[:-1], trial[-1])
                    rt = np.r_[gt, 10 * (2 * trial[ctrl] - delta)]
                    if la.norm(rt) < norm * (1 - 0.0001 * step):
                        z = trial
                        accepted = True
                        break
                step /= 2
            if not accepted:
                raise RuntimeError(
                    f"continuation failed at front={front!r} it={it!r} err={err!r}"
                )
        else:
            raise RuntimeError(f"no convergence at front={front!r}, {trace[-5:]}")
        u = np.zeros(self.K.shape[0])
        u[self.B] = z[-1] * self.ub
        u[self.V] = z[:-1]
        u[self.I] = z[-1] * self.uIunit - self.factor.solve(self.KIV @ z[:-1])
        fint = self.K @ u
        interior_error = float(np.max(abs(fint[self.I])))
        active = np.arange(len(z) - 1) != ctrl
        mineig = float(
            la.eigvalsh(H[np.ix_(active, active)], subset_by_index=[0, 0])[0]
        )
        row = dict(
            front=front,
            center=list(self.center),
            lam=float(z[-1]),
            interface_residual=float(max(abs(g))),
            bulk_free_residual=interior_error,
            opening_control=delta,
            iterations=it,
            min_controlled_tangent_eigenvalue=mineig,
            cohesive_energy=ce,
        )
        return (row, z, u)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="bicrystal", choices=["bicrystal", "polycrystal"])
    ap.add_argument("--h", type=float, default=0.5)
    ap.add_argument("--front-end", type=float, default=24)
    ap.add_argument("--front-step", type=float, default=0.5)
    args = ap.parse_args()
    tag = f"{args.case}_h{args.h:g}"
    dest = OUT / "cohesive" / tag
    dest.mkdir(parents=True, exist_ok=True)
    start = time.perf_counter()
    model = CondensedCohesive(args.case, args.h)
    save_json(dest / "verification.json", model.checks)
    print("SETUP", tag, model.checks, flush=True)
    rows = []
    z = None
    for front in np.arange(
        model.inp["a0"] + 0.5, args.front_end + 1e-07, args.front_step
    ):
        row, z, u = model.solve_front(front, z)
        rows.append(row)
        save_json(dest / "history.json", rows)
        np.savez_compressed(
            dest / "last_state.npz",
            xy=model.xy,
            conn=model.conn,
            u=u,
            interface_x=model.xv,
            opening=2 * z[:-1],
            lam=z[-1],
        )
        print(json.dumps(row), flush=True)
    save_json(
        dest / "summary.json",
        dict(
            case=args.case,
            h=args.h,
            wall_seconds=time.perf_counter() - start,
            rows=len(rows),
            lambda_min=min((r["lam"] for r in rows)),
            lambda_max=max((r["lam"] for r in rows)),
            material=model.law.report(),
            control="opening-controlled equilibrium with tip-following K boundary",
            path_status="straight cohesive plane prescribed: load comparison only",
        ),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    main()
