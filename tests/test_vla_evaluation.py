"""Prevent frozen-policy comparisons from silently changing powder physics."""
import copy
import json
import unittest

from scripts.evaluate_vla import evaluation_grain_count


class EvaluationManifestTests(unittest.TestCase):
    def setUp(self):
        self.vectors = [{"density_kg_m3": 1414.7, "radius_m": 0.0015}]
        self.manifest = {
            "legacy": False, "schema_version": "scrape_v2",
            "use": "synthetic_coarse_grain_control_data",
            "grain_count": 200, "source_version": "sha256:physical-source",
            "powder_parameter_vectors": json.dumps(self.vectors),
        }

    def validate(self, manifest):
        return evaluation_grain_count(
            manifest, source_version="sha256:physical-source", material_vectors=self.vectors,
        )

    def test_uses_top_level_recorded_count(self):
        self.manifest["dataset_attributes"] = {"grain_count": 400}
        self.assertEqual(self.validate(self.manifest), 200)

    def test_missing_or_invalid_count_never_falls_back(self):
        for value in (None, True, 0, 401, 200.0, "200"):
            with self.subTest(value=value):
                manifest = dict(self.manifest, grain_count=value)
                with self.assertRaisesRegex(ValueError, "explicit integer grain_count"):
                    self.validate(manifest)

    def test_refuses_changed_transition_sources(self):
        with self.assertRaisesRegex(ValueError, "source version"):
            self.validate(dict(self.manifest, source_version="sha256:other-source"))

    def test_refuses_changed_density_even_with_same_grain_count(self):
        vectors = copy.deepcopy(self.vectors)
        vectors[0]["density_kg_m3"] *= 1.15
        with self.assertRaisesRegex(ValueError, "material vectors"):
            self.validate(dict(self.manifest, powder_parameter_vectors=json.dumps(vectors)))

    def test_refuses_legacy_diagnostic_policy(self):
        with self.assertRaisesRegex(ValueError, "physical granular"):
            self.validate(dict(self.manifest, legacy=True, use="diagnostic_only"))


if __name__ == "__main__":
    unittest.main()
