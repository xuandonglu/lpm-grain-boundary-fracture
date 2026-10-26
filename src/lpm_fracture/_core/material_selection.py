"""Select published Mo or W elastic and surface-energy inputs.

Scientific routines extracted from the archived implementation; see provenance/source_files.json.
"""

from . import model_export as ge
from . import anisotropic_loading as ct_core
from . import grain_geometry as gb_core
from . import cleavage_calibration as sc_core
from . import cohesive_laws as uc_core

MATS = {
    "W": ((522.4, 204.4, 160.6), 3.1652e-10, {"100": 4.0, "110": 3.23}),
    "Mo": ((463.7, 157.8, 109.2), 3.147e-10, {"100": 3.18, "110": 2.8}),
}
W_C, W_A, W_GS = MATS["W"]


def use(name="W"):
    global W_C, W_A, W_GS
    W_C, W_A, W_GS = MATS[name]
    for mod in (ct_core, gb_core, sc_core, uc_core):
        if hasattr(mod, "NEP_C"):
            mod.NEP_C = W_C
        if hasattr(mod, "A_NEP"):
            mod.A_NEP = W_A
    ct_core.GS["300K"] = dict(W_GS)
    sc_core.calib_bond.__defaults__ = (W_A,)
    gb_core.KIG_A.__defaults__ = ("300K", W_C)
    init = gb_core.GBModel.__init__
    if not getattr(init, "_mat_w", False):

        def _init(self, *a, **k):
            k.setdefault("C", W_C)
            init(self, *a, **k)

        _init._mat_w = True
        gb_core.GBModel.__init__ = _init
    ge.NEP = dict(C11=W_C[0], C12=W_C[1], C44=W_C[2])
