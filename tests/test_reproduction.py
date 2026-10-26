import unittest

from lpm_fracture.data import Archive
from lpm_fracture.analysis import crossing_measure
from lpm_fracture.continuum.finite_elements import Cohesive100, inputs


class EvidenceTests(unittest.TestCase):
    def test_corrected_low_start_runs(self):
        archive = Archive()
        for tag, initiation, propagation in (
            ("sp0Rm_e19lo", 0.815, 0.935),
            ("sp0RnN_e19lo", 0.520, 0.585),
        ):
            result = crossing_measure(archive.array(tag), 16)
            self.assertAlmostEqual(result["initiation"], initiation, places=6)
            self.assertAlmostEqual(result["propagation"], propagation, places=6)

    def test_continuum_energy_limits(self):
        law = Cohesive100(inputs("bicrystal")["l"])
        self.assertAlmostEqual(float(law(0)[0]), 0, places=12)
        self.assertAlmostEqual(float(law(100)[0]) / law.Gamma, 1, places=10)

    def test_all_retained_planar_cases_present(self):
        archive = Archive()
        runs = archive.json("data/bond_softening_runs.json")
        self.assertEqual(len(runs), 36)
        self.assertEqual(sum(r["resolution"] == "coarse" for r in runs), 27)
        self.assertTrue(all((archive.root / r["path"]).is_file() for r in runs))


if __name__ == "__main__":
    unittest.main()
