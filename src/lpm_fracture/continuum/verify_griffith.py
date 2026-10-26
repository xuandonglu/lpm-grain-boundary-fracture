"""Check the independent sharp-crack FE before producing comparison data."""

import sys, json, time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
from .griffith_reference import CrackFE, run
from .finite_elements import C2, OUT, save_json


def jvals(m, state, center):
    xy, tri, u, dofs = state
    ue = u[dofs]
    strain = np.einsum("eia,ea->ei", m.B, ue)
    sig = strain @ C2
    H = 0.5 * np.einsum("ei,ei->e", strain, sig)
    ux = np.einsum("eia,ea->ei", m.grad, ue[:, ::2])
    uy = np.einsum("eia,ea->ei", m.grad, ue[:, 1::2])
    radius = np.linalg.norm(xy - center, axis=1)
    vals = []
    for r1, r2 in [(2.0, 5.0), (4.0, 8.0)]:
        q = np.clip((r2 - radius) / (r2 - r1), 0, 1)
        dq = np.einsum("eia,ea->ei", m.grad, q[tri])
        vals.append(
            float(
                np.sum(
                    m.area
                    * (
                        (sig[:, 0] * ux[:, 0] + sig[:, 2] * uy[:, 0] - H) * dq[:, 0]
                        + (sig[:, 2] * ux[:, 0] + sig[:, 1] * uy[:, 0]) * dq[:, 1]
                    )
                )
            )
        )
    return vals


def verify():
    rows = []
    for h in [1.0, 0.5, 0.25]:
        m = CrackFE("polycrystal", h)
        p = m.initial_path()
        c = m.xy[p[-1]]
        xy, t = m.split(p)
        expected = len(p) - 1
        assert len(xy) - m.nn == expected
        assert np.sum(t == p[-1]) == np.sum(
            m.tri == p[-1]
        ), "crack tip must remain connected"
        eps = np.array([0.002, 0.003, 0.001])
        u = np.column_stack(
            [
                eps[0] * m.xy[:, 0] + eps[2] / 2 * m.xy[:, 1],
                eps[1] * m.xy[:, 1] + eps[2] / 2 * m.xy[:, 0],
            ]
        ).ravel()
        dofs = (2 * m.tri[:, :, None] + np.arange(2)).reshape(-1, 6)
        energy = 0.5 * np.einsum("ea,eab,eb->", u[dofs], m.ke, u[dofs])
        exact = 0.5 * (eps @ C2 @ eps) * m.inp["W"] * m.inp["H"]
        ea, ra, sa = m.solve(p, c, all_edges=True, return_state=True)
        Ja = jvals(m, sa, c)
        ef, rf, sf = m.solve(p, c, return_state=True)
        Jf = jvals(m, sf, c)
        v = []
        for advance in [2.0, 1.0]:
            pp = m.candidate(p, 0, advance)
            ee, rr = m.solve(pp, c)
            v.append(dict(advance=advance, G=(ef - ee) / advance, force_residual=rr))
        row = dict(
            h=h,
            duplicate_face_nodes=len(xy) - m.nn,
            affine_energy_relative_error=energy / exact - 1,
            analytical_J=Ja,
            analytical_J_relative_error=[j / m.law.Gamma - 1 for j in Ja],
            actual_free_mouth_J=Jf,
            virtual_extensions=v,
            force_residual=max(ra, rf),
        )
        rows.append(row)
        print("VERIFY_GRIFFITH", json.dumps(row), flush=True)
    save_json(OUT / "griffith_verification.json", rows)
    assert abs(rows[-1]["analytical_J_relative_error"][1]) < 0.015
    assert all(
        (
            r["force_residual"] < 1e-08
            and abs(r["affine_energy_relative_error"]) < 1e-10
            for r in rows
        )
    )
    return rows


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    verify()
    for h in [1.0, 0.5]:
        run("polycrystal", h, 2.0, 106.0)
