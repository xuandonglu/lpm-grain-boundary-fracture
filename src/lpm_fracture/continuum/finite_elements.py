"""Independent continuum FE helpers for the existing Paper B transparent cases.

No LPM equilibrium, ghost interpolation, or merged connectivity is used here.
Units: GPa, current coarse lattice spacing h, energy per unit periodic thickness.
The normal cohesive potential is obtained by rigidly separating two perfect BCC
half spaces, using the archived pair AND collective potential coefficients.
This is a continuum cohesive comparison, not the identical discrete LPM model.
"""

from __future__ import annotations
import os
import hashlib
import json
from pathlib import Path
import numpy as np
import scipy.sparse as sp
from scipy.optimize import minimize_scalar, brentq

OUT = Path(os.environ.get("LPM_CONTINUUM_OUTPUT", "outputs/continuum"))
C3 = (250.7, 123.8, 98.7)
C2 = np.array([[250.7, 123.8, 0.0], [123.8, 250.7, 0.0], [0.0, 0.0, 98.7]])


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )


def inputs(case):
    from ..data import Archive

    archive = Archive()
    tag = {"bicrystal": "shRW1_e19cm025", "polycrystal": "pxtEm1_s1_ghost_e19cm025"}[
        case
    ]
    folder = (archive.root / archive.run(tag)["files"]["history"]).parent
    meta_path, kf = (folder / "model.json", folder / "model.kf")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    line = kf.read_text().splitlines()
    kval, a0, y0 = map(float, line[0].split())
    z = np.array(list(map(float, line[1].split())))
    z = z[::2] + 1j * z[1::2]
    return dict(
        case=case,
        W=meta["W"],
        H=meta["H"],
        Lz=meta["Lz"],
        a0=a0,
        y0=y0,
        l=np.array(meta["l_h"]),
        KIG=kval,
        p=z[:3],
        A=z[3:12].reshape(3, 3),
        q=z[12:],
        meta=meta,
        meta_path=str(meta_path),
        kf_path=str(kf),
    )


def kdisp(inp, xy, center):
    dx = xy[:, 0] - center[0]
    dy = xy[:, 1] - center[1]
    r, theta = (np.hypot(dx, dy), np.arctan2(dy, dx))
    z = np.cos(theta)[:, None] + np.sin(theta)[:, None] * inp["p"][None, :]
    out = np.sqrt(2 * r[:, None] / np.pi) * np.real(np.sqrt(z) * inp["q"] @ inp["A"].T)
    return out[:, :2] * inp["KIG"]


class Cohesive100:

    def __init__(self, opening_scales):
        self.l = np.asarray(opening_scales)
        self.kn = np.array([0.75 * C3[2], 0.25 * (C3[0] - C3[1])])
        self.T = 0.75 * (C3[1] - C3[2]) / 14
        self.E = 2 * self.kn * self.l**2
        self.Gamma = float(4 * self.E[0] + 2 * self.E[1])
        peak = minimize_scalar(
            lambda d: -float(self(d)[1]),
            bounds=(0, 10 * max(self.l)),
            method="bounded",
            options={"xatol": 1e-13},
        )
        self.dpeak, self.tpeak = (float(peak.x), float(-peak.fun))
        self.d90 = float(
            brentq(
                lambda d: self(d)[0] / self.Gamma - 0.9, self.dpeak, 50 * max(self.l)
            )
        )
        self.Eeffective = C3[0] - C3[1] ** 2 / C3[0]
        self.ell_strength_AT2 = (
            27 * self.Eeffective * self.Gamma / (256 * self.tpeak**2)
        )

    def __call__(self, delta):
        d = np.asarray(delta, float)
        r = np.sqrt((0.5 + d) ** 2 + 0.5)
        x = np.stack([r - np.sqrt(0.75), d], axis=-1)
        xp = np.stack([(0.5 + d) / r, np.ones_like(d)], axis=-1)
        xpp = np.stack([0.5 / r**3, np.zeros_like(d)], axis=-1)
        rat = x / self.l
        e = np.exp(-rat)
        phi = self.E * (1 - (1 + rat) * e)
        phip = 2 * self.kn * x * e
        phipp = 2 * self.kn * (1 - rat) * e
        psi = x * e
        psip = (1 - rat) * e
        psipp = (rat - 2) * e / self.l
        count = np.array([4.0, 2.0])
        collective = np.array([16.0, 2.0]) * self.T
        energy = np.sum(count * phi + collective * psi**2, axis=-1)
        traction = np.sum((count * phip + 2 * collective * psi * psip) * xp, axis=-1)
        tangent = np.sum(
            (count * phipp + 2 * collective * (psip**2 + psi * psipp)) * xp**2
            + (count * phip + 2 * collective * psi * psip) * xpp,
            axis=-1,
        )
        return (energy, traction, tangent)

    def report(self):
        return dict(
            C_GPa=C2.tolist(),
            opening_scales_h=self.l.tolist(),
            Kn=self.kn.tolist(),
            T=float(self.T),
            shell_energies=self.E.tolist(),
            G100_GPa_h=self.Gamma,
            G110_GPa_h=float(2 * np.sqrt(2) * (self.E[0] + self.E[1])),
            delta_peak_h=self.dpeak,
            traction_peak_GPa=self.tpeak,
            delta_90_h=self.d90,
            AT2_strength_matched_ell_h=self.ell_strength_AT2,
            definition="Exact finite-opening rigid-slab normal potential, including the collective term; linear anisotropic continuum bulk; not a discrete lattice equilibrium.",
        )


def grid(xs, ys):
    xx, yy = np.meshgrid(xs, ys)
    xy = np.column_stack([xx.ravel(), yy.ravel()])
    nx, ny = (len(xs) - 1, len(ys) - 1)
    ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
    base = (jj * (nx + 1) + ii).ravel()
    conn = np.column_stack([base, base + 1, base + nx + 2, base + nx + 1])
    dofs = (2 * conn[:, :, None] + np.arange(2)).reshape(-1, 8)
    return (xy, conn, dofs)


def quadrature(xy, conn):
    gp = np.array([-1, 1]) / np.sqrt(3)
    Ns, grads, weights, Bs = ([], [], [], [])
    hx = xy[conn[:, 1], 0] - xy[conn[:, 0], 0]
    hy = xy[conn[:, 3], 1] - xy[conn[:, 0], 1]
    for xi in gp:
        for eta in gp:
            N = 0.25 * np.array(
                [
                    (1 - xi) * (1 - eta),
                    (1 + xi) * (1 - eta),
                    (1 + xi) * (1 + eta),
                    (1 - xi) * (1 + eta),
                ]
            )
            dx = (
                0.5
                * np.array([-(1 - eta), 1 - eta, 1 + eta, -(1 + eta)])[None, :]
                / hx[:, None]
            )
            dy = (
                0.5
                * np.array([-(1 - xi), -(1 + xi), 1 + xi, 1 - xi])[None, :]
                / hy[:, None]
            )
            grad = np.stack([dx, dy], axis=1)
            B = np.zeros((len(conn), 3, 8))
            B[:, 0, ::2] = dx
            B[:, 1, 1::2] = dy
            B[:, 2, ::2] = dy
            B[:, 2, 1::2] = dx
            Ns.append(N)
            grads.append(grad)
            weights.append(hx * hy / 4)
            Bs.append(B)
    return (np.array(Ns), np.array(grads), np.array(weights), np.array(Bs))


def stiffness(xy, conn, dofs, C=C2, degradation=None):
    Ns, grads, weights, Bs = quadrature(xy, conn)
    ke = np.zeros((len(conn), 8, 8))
    for g, (B, w) in enumerate(zip(Bs, weights)):
        fac = w if degradation is None else w * degradation[g]
        ke += np.einsum("eia,ij,ejb,e->eab", B, C, B, fac, optimize=True)
    rows = np.repeat(dofs, 8, axis=1).ravel()
    cols = np.tile(dofs, 8).ravel()
    return sp.coo_matrix(
        (ke.ravel(), (rows, cols)), shape=(2 * len(xy), 2 * len(xy))
    ).tocsr()


def material_checks():
    out = {}
    for case in ["bicrystal", "polycrystal"]:
        inp = inputs(case)
        law = Cohesive100(inp["l"])
        d = np.array([0.001, 0.01, 0.04, 0.1, 0.3])
        eps = 1e-06
        energy, tr, tang = law(d)
        fd1 = (law(d + eps)[0] - law(d - eps)[0]) / (2 * eps)
        fd2 = (law(d + eps)[1] - law(d - eps)[1]) / (2 * eps)
        report = law.report()
        report["energy_derivative_relative_error"] = float(
            np.linalg.norm(fd1 - tr) / np.linalg.norm(tr)
        )
        report["traction_derivative_relative_error"] = float(
            np.linalg.norm(fd2 - tang) / np.linalg.norm(tang)
        )
        report["asymptotic_energy_relative_error"] = abs(
            float(law(100 * max(inp["l"]))[0]) / law.Gamma - 1
        )
        report["geometry"] = {k: inp[k] for k in ["W", "H", "a0", "y0", "Lz", "KIG"]}
        report["sources"] = {p: sha(p) for p in [inp["meta_path"], inp["kf_path"]]}
        out[case] = report
    save_json(OUT / "material_mapping.json", out)
    return out


if __name__ == "__main__":
    import sys

    sys.stdout.reconfigure(encoding="utf-8")
    print(json.dumps(material_checks(), indent=2))
