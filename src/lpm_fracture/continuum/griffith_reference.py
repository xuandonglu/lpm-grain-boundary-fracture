"""Independent FE virtual-crack-extension comparison, exact directional work.

Crossed CST triangulation supplies both diagonal cleavage directions without
prescribing the future crack. Displacement sectors are split only along crack
edges; all intact faces retain ordinary continuum continuity. At each step all
forward {100}/{110} candidates are solved at the SAME unit boundary field.
The candidate with smallest Gamma*new_length/(elastic energy released) grows.
This is a sharp-crack Griffith comparison, not a bond-cohesive error oracle.
"""

from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sla
from .finite_elements import inputs, Cohesive100, C2, kdisp, OUT, save_json


class CrackFE:

    def __init__(self, case, h):
        self.inp = inp = inputs(case)
        self.law = Cohesive100(inp["l"])
        self.h = h
        self.nx = nx = round(inp["W"] / h)
        self.ny = ny = round(inp["H"] / h)
        xs = np.linspace(0, inp["W"], nx + 1)
        ys = np.linspace(inp["y0"] - inp["H"] / 2, inp["y0"] + inp["H"] / 2, ny + 1)
        xx, yy = np.meshgrid(xs, ys)
        corners = np.column_stack([xx.ravel(), yy.ravel()])
        self.ncorner = len(corners)
        ii, jj = np.meshgrid(np.arange(nx), np.arange(ny))
        base = (jj * (nx + 1) + ii).ravel()
        centers = np.column_stack(
            [(xs[:-1] + h / 2)[ii.ravel()], (ys[:-1] + h / 2)[jj.ravel()]]
        )
        self.xy = np.vstack([corners, centers])
        self.nn = len(self.xy)
        c = np.arange(len(base)) + self.ncorner
        tris = np.stack(
            [
                np.column_stack([base, base + 1, c]),
                np.column_stack([base + 1, base + nx + 2, c]),
                np.column_stack([base + nx + 2, base + nx + 1, c]),
                np.column_stack([base + nx + 1, base, c]),
            ],
            axis=1,
        ).reshape(-1, 3)
        self.tri = tris
        x = self.xy[tris, 0]
        y = self.xy[tris, 1]
        twice = (x[:, 1] - x[:, 0]) * (y[:, 2] - y[:, 0]) - (x[:, 2] - x[:, 0]) * (
            y[:, 1] - y[:, 0]
        )
        assert np.min(twice) > 0
        self.area = twice / 2
        dx = (
            np.column_stack([y[:, 1] - y[:, 2], y[:, 2] - y[:, 0], y[:, 0] - y[:, 1]])
            / twice[:, None]
        )
        dy = (
            np.column_stack([x[:, 2] - x[:, 1], x[:, 0] - x[:, 2], x[:, 1] - x[:, 0]])
            / twice[:, None]
        )
        self.grad = np.stack([dx, dy], axis=1)
        self.B = np.zeros((len(tris), 3, 6))
        self.B[:, 0, ::2] = dx
        self.B[:, 1, 1::2] = dy
        self.B[:, 2, ::2] = dy
        self.B[:, 2, 1::2] = dx
        self.ke = np.einsum(
            "eia,ij,ejb,e->eab", self.B, C2, self.B, self.area, optimize=True
        )
        self.inc_order = np.argsort(tris.ravel(), kind="stable")
        self.inc_ptr = np.r_[0, np.cumsum(np.bincount(tris.ravel(), minlength=self.nn))]
        self.edge_set = set()
        for a, b in [(0, 1), (1, 2), (2, 0)]:
            pairs = np.sort(tris[:, [a, b]], axis=1)
            self.edge_set.update(map(tuple, pairs.tolist()))

    def corner(self, x, y):
        i = round(x / self.h)
        j = round((y - (self.inp["y0"] - self.inp["H"] / 2)) / self.h)
        if not (0 <= i <= self.nx and 0 <= j <= self.ny):
            raise ValueError("candidate outside mesh")
        return j * (self.nx + 1) + i

    def initial_path(self):
        j = self.ny // 2
        return [
            j * (self.nx + 1) + i for i in range(round(self.inp["a0"] / self.h) + 1)
        ]

    def candidate(self, path, angle, advance):
        x, y = self.xy[path[-1]]
        n = max(1, round(advance / self.h))
        nodes = []
        for k in range(n):
            if angle == 0:
                x += self.h
                nodes.append(self.corner(x, y))
            elif abs(angle) == 90:
                y += np.sign(angle) * self.h
                nodes.append(self.corner(x, y))
            else:
                sign = np.sign(angle)
                ix = round(x / self.h)
                jy = round((y - (self.inp["y0"] - self.inp["H"] / 2)) / self.h)
                cell_j = jy if sign > 0 else jy - 1
                if not (0 <= ix < self.nx and 0 <= cell_j < self.ny):
                    raise ValueError("candidate outside mesh")
                nodes.append(self.ncorner + cell_j * self.nx + ix)
                x += self.h
                y += sign * self.h
                nodes.append(self.corner(x, y))
        return path + nodes

    def split(self, path):
        cut = {tuple(sorted((a, b))) for a, b in zip(path[:-1], path[1:])}
        assert cut <= self.edge_set, "crack does not follow actual FE edges"
        tri = self.tri.copy()
        coords = [self.xy]
        added = []
        newid = self.nn
        for vertex in set(path):
            occur = self.inc_order[self.inc_ptr[vertex] : self.inc_ptr[vertex + 1]]
            ts = occur // 3
            pos = occur % 3
            parent = np.arange(len(ts))

            def find(a):
                while parent[a] != a:
                    parent[a] = parent[parent[a]]
                    a = parent[a]
                return a

            across = {}
            for loc, t in enumerate(ts):
                for other in self.tri[t]:
                    if other != vertex:
                        across.setdefault(int(other), []).append(loc)
            for other, incident in across.items():
                if len(incident) == 2 and tuple(sorted((vertex, other))) not in cut:
                    p, q = map(find, incident)
                    parent[p] = q
            roots = {}
            for loc, t in enumerate(ts):
                root = find(loc)
                if root not in roots:
                    if not roots:
                        roots[root] = vertex
                    else:
                        roots[root] = newid
                        newid += 1
                        added.append(self.xy[vertex])
                tri[t, pos[loc]] = roots[root]
        if added:
            coords.append(np.array(added))
        return (np.vstack(coords), tri)

    def solve(self, path, center, all_edges=False, return_state=False):
        xy, tri = self.split(path)
        dofs = (2 * tri[:, :, None] + np.arange(2)).reshape(-1, 6)
        rows = np.repeat(dofs, 6, axis=1).ravel()
        cols = np.tile(dofs, 6).ravel()
        K = sp.coo_matrix(
            (self.ke.ravel(), (rows, cols)), shape=(2 * len(xy), 2 * len(xy))
        ).tocsr()
        i = self.inp
        dbc = np.minimum.reduce(
            [
                i["W"] - xy[:, 0],
                xy[:, 1] - (i["y0"] - i["H"] / 2),
                i["y0"] + i["H"] / 2 - xy[:, 1],
            ]
        )
        if all_edges:
            dbc = np.minimum(dbc, xy[:, 0])
        fixed = np.flatnonzero(np.repeat(dbc < 1.5 - 1e-08, 2))
        free = np.setdiff1d(np.arange(2 * len(xy)), fixed)
        at = xy[fixed // 2].copy()
        if all_edges:
            for node in np.unique(fixed // 2):
                if abs(xy[node, 1] - i["y0"]) < 1e-08 and xy[node, 0] < center[0]:
                    ts = np.flatnonzero(np.any(tri == node, axis=1))
                    sy = np.sign(np.mean(self.xy[self.tri[ts], 1]) - i["y0"])
                    at[fixed // 2 == node, 1] += sy * 1e-12
        vals = kdisp(i, at, center)
        u = np.zeros(2 * len(xy))
        u[fixed] = vals[np.arange(len(fixed)), fixed % 2]
        u[free] = sla.spsolve(K[free][:, free].tocsc(), -K[free][:, fixed] @ u[fixed])
        force = K @ u
        energy = float(0.5 * u @ force)
        err = float(max(abs(force[free])))
        if return_state:
            return (energy, err, (xy, tri, u, dofs))
        return (energy, err)

    def gamma(self, angle):
        a = np.radians(angle)
        n = np.array([-np.sin(a), np.cos(a)])
        E1, E2 = self.law.E
        return float(
            2 * E1 * (abs(n[0] + n[1]) + abs(n[0] - n[1])) + 2 * E2 * np.sum(abs(n))
        )

    def step(self, path, advance):
        center = self.xy[path[-1]].copy()
        base, err = self.solve(path, center)
        rows = []
        for angle in [-90, -45, 0, 45, 90]:
            try:
                trial = self.candidate(path, angle, advance)
            except ValueError:
                continue
            en, re = self.solve(trial, center)
            delta = np.diff(self.xy[trial[len(path) - 1 :]], axis=0)
            length = float(np.sum(np.linalg.norm(delta, axis=1)))
            released = base - en
            crit = (
                np.sqrt(self.gamma(angle) * length / released)
                if released > 0
                else np.inf
            )
            rows.append(
                dict(
                    angle=angle,
                    new_length=length,
                    energy_release_unit=base - en,
                    G_unit=released / length,
                    Gamma=self.gamma(angle),
                    critical_lambda=float(crit) if np.isfinite(crit) else None,
                    free_force_residual=max(err, re),
                    end=self.xy[trial[-1]].tolist(),
                )
            )
        valid = [r for r in rows if r["critical_lambda"] is not None]
        chosen = min(valid, key=lambda r: r["critical_lambda"])
        return (self.candidate(path, chosen["angle"], advance), chosen, rows)


def run(case, h, advance, tip_end, resume=False):
    tag = f"{case}_h{h:g}_da{advance:g}"
    dest = OUT / "griffith" / tag
    dest.mkdir(parents=True, exist_ok=True)
    m = CrackFE(case, h)
    path = m.initial_path()
    start = time.perf_counter()
    history = []
    directions = []
    if resume and (dest / "history.json").exists():
        history = json.loads((dest / "history.json").read_text())
        directions = json.loads((dest / "candidate_comparisons.json").read_text())
        path = np.load(dest / "path.npz")["node_ids"].tolist()
        assert np.allclose(m.xy[path[-1]], history[-1]["end"])
    for step in range(len(history), 200):
        if m.xy[path[-1], 0] >= tip_end - 1e-08:
            break
        old = m.xy[path[-1]].tolist()
        path, row, candidates = m.step(path, advance)
        row.update(
            step=step + 1,
            start=old,
            front=float(m.xy[path[-1], 0]),
            tipy=float(m.xy[path[-1], 1]),
        )
        history.append(row)
        directions.append(candidates)
        save_json(dest / "history.json", history)
        save_json(dest / "candidate_comparisons.json", directions)
        np.savez_compressed(dest / "path.npz", xy=m.xy[path], node_ids=np.array(path))
        print("GRIFFITH", tag, json.dumps(row), flush=True)
        if row["front"] >= tip_end - 1e-08:
            break
        if abs(row["tipy"] - m.inp["y0"]) > m.inp["H"] / 2 - 7:
            break
    en, err, state = m.solve(path, m.xy[path[-1]], return_state=True)
    xy, tri, u, dofs = state
    np.savez_compressed(dest / "last_state.npz", xy=xy, tri=tri, u=u, path=m.xy[path])
    save_json(
        dest / "summary.json",
        dict(
            case=case,
            h_FE=h,
            advance=advance,
            steps=len(history),
            final_tip=history[-1]["front"],
            all_chosen_angles=[r["angle"] for r in history],
            wall_seconds=time.perf_counter() - start,
            final_force_residual=err,
            status=(
                "reached_target"
                if history[-1]["front"] >= tip_end - 1e-08
                else "stopped"
            ),
            model="sharp-crack Griffith energy with independent conforming CST finite elements",
            gamma_definition="same exact two-shell broken-bond directional work; no load or path fitted",
        ),
    )


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", default="polycrystal")
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--advance", type=float, default=2.0)
    ap.add_argument("--tip-end", type=float, default=118.0)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    run(args.case, args.h, args.advance, args.tip_end, args.resume)
