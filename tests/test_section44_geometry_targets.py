import unittest

import numpy as np

from paper_cvrl.section44_geometry_targets import local_geometry_targets


class TestGeometryTargets(unittest.TestCase):
    def test_plane_line_and_scale(self):
        grid = np.linspace(-2, 2, 21)
        xx, yy = np.meshgrid(grid, grid)
        plane = np.column_stack((xx.ravel(), yy.ravel(), np.zeros(xx.size)))
        flat = local_geometry_targets(plane)
        self.assertTrue(flat.valid.all())
        self.assertLess(float(np.quantile(flat.curvature, 0.95)), 1e-5)
        self.assertGreater(float(np.median(flat.planarity)), 0.2)
        scaled = local_geometry_targets(plane * 8)
        np.testing.assert_allclose(flat.curvature, scaled.curvature, atol=1e-6)
        np.testing.assert_allclose(flat.planarity, scaled.planarity, atol=1e-6)
        line = np.column_stack((np.linspace(-5, 5, 200), np.zeros(200), np.zeros(200)))
        linear = local_geometry_targets(line)
        self.assertLess(float(np.max(linear.planarity)), 1e-5)

    def test_curved_surface_and_degenerate(self):
        rng = np.random.default_rng(7)
        sphere = rng.normal(size=(1200, 3))
        sphere /= np.linalg.norm(sphere, axis=1, keepdims=True)
        curved = local_geometry_targets(sphere, k=64)
        self.assertTrue(curved.valid.all())
        self.assertGreater(float(np.median(curved.curvature)), 0)
        coincident = local_geometry_targets(np.zeros((64, 3)))
        self.assertFalse(coincident.valid.any())
        self.assertTrue(np.isfinite(coincident.curvature).all())


if __name__ == "__main__":
    unittest.main()
