"""Binary model.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

import numpy as np


def read(path):
    with open(path, "rb") as f:
        h = np.fromfile(f, np.int64, 8)
        N, nb, nsup, ver = (int(h[0]), int(h[1]), int(h[2]), int(h[3]))
        hd = np.fromfile(f, float, 16 if ver == 2 else 8)
        d = {"N": N, "nb": nb, "nsup": nsup, "ver": ver, "hdr": hd}
        d["X"] = np.fromfile(f, float, 3 * N).reshape(N, 3)
        d["fixed"] = np.fromfile(f, np.uint8, 3 * N).astype(bool)
        d["ubc"] = np.fromfile(f, float, 3 * N)
        d["owner"] = np.fromfile(f, np.int32, nb)
        d["shell"] = np.fromfile(f, np.int32, nb)
        d["pair"] = np.fromfile(f, np.int32, nb)
        d["flag"] = np.fromfile(f, np.uint8, nb)
        d["n"] = np.fromfile(f, float, 3 * nb).reshape(nb, 3)
        d["d0"] = np.fromfile(f, float, nb)
        d["kn"] = np.fromfile(f, float, nb)
        d["xi"] = np.fromfile(f, float, nb)
        d["sptr"] = np.fromfile(f, np.int64, nb + 1)
        d["sidx"] = np.fromfile(f, np.int32, nsup)
        d["sw"] = np.fromfile(f, float, nsup)
        if ver == 2:
            d["coh"] = np.fromfile(f, np.uint8, nb).astype(bool)
            d["gmul"] = np.fromfile(f, float, nb)
    return d
